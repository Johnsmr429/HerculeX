#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <vecLib/vDSP.h>

#define SLOTS 3

const char *lane_kernels = "ink/model/reader_apple/reader.metal";

typedef _Float16 f16;

static void *apple_fft_open(int side) {
    return side > 1 && !(side & (side - 1))
        ? vDSP_create_fftsetupD(__builtin_ctz((u32)side), kFFTRadix2)
        : NULL;
}

static void apple_fft_rows(void *plan, f64 *re, f64 *im, int side, int rows, int inverse) {
    DSPDoubleSplitComplex data = {re, im};
    vDSP_fftm_zipD(
        plan, &data, 1, side, __builtin_ctz((u32)side), rows, inverse ? FFT_INVERSE : FFT_FORWARD);
}

static void apple_fft_close(void *plan) { vDSP_destroy_fftsetupD(plan); }

typedef struct {
    u32 height, width;
    f32 attention_bias;
} StemDims;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> stem, first, second;
    MLModel *engine;
    MPSGraph *graph;
    MPSGraphTensor *body_in, *body_out;
    Stem stem_weights;
    Scale scale;
    id<MTLBuffer> table, w1, b1, w2, b2, att;
    id<MTLBuffer> tiles, conv1, conv2, measured1, measured2, norm1, measure_table;
    u32 *bins;
    pthread_mutex_t measuring;
    id<MTLBuffer> block[SLOTS], features[SLOTS];
    f32 *reads[SLOTS];
    dispatch_semaphore_t done[SLOTS];
    int failed[SLOTS];
    Reading work;
    f32 *weight, *sigmoid;
    struct Flow *flows;
    int slot_flow[SLOTS];
    Place slot_place[SLOTS];
    i64 issued, settled, queued, finished;
} Reader;

#define FLOWS (SLOTS + 1)

typedef struct Flow {
    f32 *total;
    int h, w, ty, tx, count, settled;
    Place places[4096];
    f32 blend[TILE * TILE];
} Flow;

void copy_half(f16 *to, Tensor *t, i64 n) {
    for (i64 i = 0; i < n; i++) {
        to[i] = t->data[i];
    }
}

id<MTLComputePipelineState> pipeline_of(
    id<MTLDevice> device, id<MTLLibrary> library, const char *name) {
    NSError *error = nil;
    id<MTLFunction> f = [library newFunctionWithName:@(name)];
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&error];
    need(p != nil, "metal %s: %s", name, error.localizedDescription.UTF8String);
    return p;
}

MLModel *engine_open(const char *compiled) {
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@(compiled)]
                                       configuration:configuration
                                               error:&error];
    need(model != nil, "engine %s: %s", compiled, error.localizedDescription.UTF8String);
    return model;
}

typedef struct {
    MPSGraph *g;
    Tensor *tensors;
    int count;
    Arena *a;
} Body;

MPSGraphTensor *constant_of(Body *b, const char *name, NSArray<NSNumber *> *shape) {
    Tensor *t = tensor_named(b->tensors, b->count, fmt(b->a, "model.network.%s", name));
    i64 n = tensor_elements(t);
    NSMutableData *data = [NSMutableData dataWithLength:n * 2];
    copy_half(data.mutableBytes, t, n);
    if (!shape) {
        NSMutableArray *dims = [NSMutableArray array];
        for (int d = 0; d < t->ndim; d++) {
            [dims addObject:@(t->shape[d])];
        }
        shape = dims;
    }
    return [b->g constantWithData:data shape:shape dataType:MPSDataTypeFloat16];
}

MPSGraphTensor *channels_of(Body *b, const char *name, int channels) {
    return constant_of(b, name, @[ @1, @(channels), @1, @1 ]);
}

MPSGraphTensor *conv(
    Body *b, MPSGraphTensor *x, const char *prefix, int out, int stride, int pad, int bias) {
    MPSGraphConvolution2DOpDescriptor *d = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:stride
                      strideInY:stride
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                    paddingLeft:pad
                   paddingRight:pad
                     paddingTop:pad
                  paddingBottom:pad
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    MPSGraphTensor *y =
        [b->g convolution2DWithSourceTensor:x
                              weightsTensor:constant_of(b, fmt(b->a, "%s.weight", prefix), nil)
                                 descriptor:d
                                       name:nil];
    if (!bias) {
        return y;
    }
    return [b->g additionWithPrimaryTensor:y
                           secondaryTensor:channels_of(b, fmt(b->a, "%s.bias", prefix), out)
                                      name:nil];
}

MPSGraphTensor *norm(Body *b, MPSGraphTensor *x, const char *prefix, int channels) {
    MPSGraphTensor *f = [b->g castTensor:x toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *mean = [b->g meanOfTensor:f axes:@[ @2, @3 ] name:nil];
    MPSGraphTensor *variance = [b->g varianceOfTensor:f meanTensor:mean axes:@[ @2, @3 ] name:nil];
    MPSGraphTensor *gamma = channels_of(b, fmt(b->a, "%s.weight", prefix), channels);
    MPSGraphTensor *beta = channels_of(b, fmt(b->a, "%s.bias", prefix), channels);
    MPSGraphTensor *y =
        [b->g normalizationWithTensor:f
                           meanTensor:mean
                       varianceTensor:variance
                          gammaTensor:[b->g castTensor:gamma toType:MPSDataTypeFloat32 name:nil]
                           betaTensor:[b->g castTensor:beta toType:MPSDataTypeFloat32 name:nil]
                              epsilon:1e-5
                                 name:nil];
    return [b->g castTensor:y toType:MPSDataTypeFloat16 name:nil];
}

MPSGraphTensor *leaky(Body *b, MPSGraphTensor *x) {
    return [b->g leakyReLUWithTensor:x alpha:0.01 name:nil];
}

MPSGraphTensor *conv_norm(
    Body *b, MPSGraphTensor *x, const char *prefix, int out, int stride, int pad) {
    MPSGraphTensor *y = conv(b, x, fmt(b->a, "%s.conv", prefix), out, stride, pad, pad > 0);
    return norm(b, y, fmt(b->a, "%s.norm", prefix), out);
}

void body_graph(Reader *r, Arena *a, Tensor *tensors, int count) {
    Body b = {[MPSGraph new], tensors, count, a};
    int widths[6] = {32, 64, 128, 256, 320, 320}, blocks[6] = {1, 3, 4, 6, 6, 6};
    r->graph = b.g;
    r->body_in = [b.g placeholderWithShape:@[ @1, @32, @-1, @-1 ]
                                  dataType:MPSDataTypeFloat16
                                      name:nil];
    MPSGraphTensor *x = conv_norm(&b, r->body_in, "shared_encoder.stem.convs.0", 32, 1, 1);
    x = leaky(&b, x);
    MPSGraphTensor *skips[6];
    int width = 32;
    MPSGraphPooling2DOpDescriptor *pool =
        [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:2
                                                    kernelHeight:2
                                                       strideInX:2
                                                       strideInY:2
                                                    paddingStyle:MPSGraphPaddingStyleTF_VALID
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
    for (int s = 0; s < 6; s++) {
        for (int k = 0; k < blocks[s]; k++) {
            const char *block = fmt(a, "shared_encoder.stages.%d.blocks.%d", s, k);
            int stride = s > 0 && k == 0 ? 2 : 1;
            MPSGraphTensor *skip = x;
            if (stride == 2) {
                skip = [b.g avgPooling2DWithSourceTensor:skip descriptor:pool name:nil];
            }
            if (width != widths[s]) {
                skip = conv_norm(&b, skip, fmt(a, "%s.skip.1", block), widths[s], 1, 0);
            }
            MPSGraphTensor *y =
                leaky(&b, conv_norm(&b, x, fmt(a, "%s.conv1", block), widths[s], stride, 1));
            y = conv_norm(&b, y, fmt(a, "%s.conv2", block), widths[s], 1, 1);
            x = leaky(&b, [b.g additionWithPrimaryTensor:y secondaryTensor:skip name:nil]);
            width = widths[s];
        }
        skips[s] = x;
    }
    MPSGraphConvolution2DOpDescriptor *up = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:2
                      strideInY:2
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                    paddingLeft:0
                   paddingRight:0
                     paddingTop:0
                  paddingBottom:0
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    for (int s = 0; s < 5; s++) {
        MPSGraphTensor *skip = skips[4 - s];
        int out = widths[4 - s];
        MPSGraphTensor *y = [b.g
            convolutionTranspose2DWithSourceTensor:x
                                     weightsTensor:constant_of(&b,
                                                       fmt(a,
                                                           "shared_decoder.transpconvs.%d.weight",
                                                           s),
                                                       nil)
                                 outputShapeTensor:[b.g shapeOfTensor:skip name:nil]
                                        descriptor:up
                                              name:nil];
        y = [b.g additionWithPrimaryTensor:y
                           secondaryTensor:channels_of(&b,
                                               fmt(a, "shared_decoder.transpconvs.%d.bias", s), out)
                                      name:nil];
        y = [b.g concatTensors:@[ y, skip ] dimension:1 name:nil];
        const char *stage = fmt(a, "shared_decoder.stages.%d.convs.0", s);
        y = conv(&b, y, fmt(a, "%s.conv", stage), out, 1, 1, 1);
        x = leaky(&b, norm(&b, y, fmt(a, "%s.norm", stage), out));
    }
    r->body_out = conv(&b, x, "task_heads.ink", 1, 1, 0, 1);
}

void body_tile(Reader *r, int slot, int ty, int tx) {
    MPSGraphTensorData *input =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:r->features[slot]
                                                shape:@[ @1, @32, @(ty), @(tx) ]
                                             dataType:MPSDataTypeFloat16];
    NSDictionary *result = [r->graph runWithMTLCommandQueue:r->queue
                                                      feeds:@{r->body_in : input}
                                              targetTensors:@[ r->body_out ]
                                           targetOperations:nil];
    u16 *logits = r->block[slot].contents;
    MPSGraphTensorData *out = result[r->body_out];
    [out.mpsndarray readBytes:logits strideBytes:nil];
    for (int i = 0; i < ty * tx; i++) {
        r->reads[slot][i] = r->sigmoid[logits[i]];
    }
    r->failed[slot] = 0;
    dispatch_semaphore_signal(r->done[slot]);
}

Reader *reader_open(Arena *a, const char *weights, i64 columns) {
    const char *engines = fmt(a, "%s/weights/engine", LIBRARY);
    const char *kernels = fmt(a, "%s/ink/model/reader_apple/reader.metal", HOME);
    Reader *r = pushz(a, sizeof(Reader));
    reading_init(&r->work, a, columns);
    r->weight = push(a, columns * sizeof(f32));
    r->sigmoid = push(a, 65536 * sizeof(*r->sigmoid));
    for (u32 i = 0; i < 65536; i++) {
        u16 bits = i;
        f16 logit;
        memcpy(&logit, &bits, sizeof(logit));
        f32 x = logit;
        r->sigmoid[i] = x < -100 ? 0 : x > 100 ? 1 : 1 / (1 + expf(-x));
    }
    r->flows = pushz(a, FLOWS * sizeof(Flow));
    Arena checkpoint = arena(a->size);
    Tensor *tensors = pushz(&checkpoint, TENSORS * sizeof(Tensor));
    int count = load_tensors(&checkpoint, weights, tensors);
    r->stem_weights = stem_of(&checkpoint, tensors, count);
    f32 *all = (f32 *)&r->stem_weights;
    for (u64 k = 0; k < sizeof(Stem) / sizeof(f32); k++) {
        all[k] = (f16)all[k];
    }
    body_graph(r, &checkpoint, tensors, count);
    munmap(checkpoint.base, checkpoint.size);
    EVP_MD_CTX *hash = EVP_MD_CTX_new();
    need(hash && EVP_DigestInit_ex(hash, EVP_sha256(), NULL), "cannot start checkpoint hash");
    ink_hash_file(hash, weights);
    u8 d[32];
    need(EVP_DigestFinal_ex(hash, d, NULL), "cannot finish checkpoint hash");
    EVP_MD_CTX_free(hash);
    char hex[17];
    for (int i = 0; i < 8; i++) {
        snprintf(hex + 2 * i, 3, "%02x", d[i]);
    }
    const char *compiled = fmt(a, "%s/%s-%d.mlmodelc", engines, hex, TILE);
    r->engine = exists(compiled) ? engine_open(compiled) : nil;
    if (!r->engine) {
        fprintf(stderr, "no Neural Engine body at %s; every tile is read on the GPU\n", compiled);
    }
    r->device = MTLCreateSystemDefaultDevice();
    r->queue = [r->device newCommandQueue];
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    Bytes source = read_file(a, kernels);
    id<MTLLibrary> library =
        [r->device newLibraryWithSource:[[NSString alloc] initWithBytes:source.data
                                                                 length:source.size
                                                               encoding:NSUTF8StringEncoding]
                                options:options
                                  error:&error];
    need(library != nil, "%s: %s", kernels, error.localizedDescription.UTF8String);
    r->stem = pipeline_of(r->device, library, "depth_stem");
    r->first = pipeline_of(r->device, library, "stem_first");
    r->second = pipeline_of(r->device, library, "stem_second");
    MTLResourceOptions shared = MTLResourceStorageModeShared;
    i64 measured = (i64)MEASURED_TILES * DEPTH * MEASURED_SIDE * MEASURED_SIDE;
    r->table = [r->device newBufferWithLength:256 * 2 options:shared];
    r->w1 = [r->device newBufferWithLength:27 * STEM * 2 options:shared];
    r->b1 = [r->device newBufferWithLength:STEM * 2 options:shared];
    r->w2 = [r->device newBufferWithLength:27 * STEM * STEM * 2 options:shared];
    r->b2 = [r->device newBufferWithLength:STEM * 2 options:shared];
    r->att = [r->device newBufferWithLength:STEM * 2 options:shared];
    r->tiles = [r->device newBufferWithLength:measured options:shared];
    r->norm1 = [r->device newBufferWithLength:2 * STEM * 2 options:shared];
    r->measured1 = [r->device newBufferWithLength:measured * STEM * 2 options:shared];
    r->measured2 = [r->device newBufferWithLength:measured * STEM * 2 options:shared];
    r->measure_table = [r->device newBufferWithLength:256 * 2 options:shared];
    pthread_mutex_init(&r->measuring, 0);
    r->bins = push(a, STEM * 65536 * 4);
    f16 *attention = r->att.contents;
    for (int c = 0; c < STEM; c++) {
        attention[c] = r->stem_weights.attention[c];
    }
    r->conv1 = [r->device newBufferWithLength:STEM * 27 * 2 options:shared];
    r->conv2 = [r->device newBufferWithLength:STEM * STEM * 27 * 2 options:shared];
    f16 *conv1 = r->conv1.contents, *conv2 = r->conv2.contents;
    for (int i = 0; i < STEM * 27; i++) {
        conv1[i] = r->stem_weights.conv1[i];
    }
    for (int tap = 0; tap < 27; tap++) {
        for (int in = 0; in < STEM; in++) {
            for (int out = 0; out < STEM; out++) {
                conv2[(tap * STEM + in) * STEM + out] =
                    r->stem_weights.conv2[(out * STEM + in) * 27 + tap];
            }
        }
    }
    for (int s = 0; s < SLOTS; s++) {
        r->block[s] = [r->device newBufferWithLength:DEPTH * TILE * TILE options:shared];
        r->features[s] = [r->device newBufferWithLength:2 * STEM * TILE * TILE * 2 options:shared];
        r->reads[s] = push(a, TILE * TILE * sizeof(f32));
        r->done[s] = dispatch_semaphore_create(0);
    }
    return r;
}

typedef struct {
    i32 tiles, height, width;
    u32 count;
} Box;

typedef struct {
    const u16 *x;
    i64 tile_size;
    int tiles;
    const int *frequency;
    u32 *bins;
    f32 *mean, *variance;
} Moments;

void channel_moments_of(void *context, i64 c) {
    Moments *m = context;
    u32 *counts = m->bins + c * 65536;
    const u16 *x = m->x + c * m->tiles * m->tile_size;
    memset(counts, 0, 65536 * 4);
    for (int k = 0; k < m->tiles; k++, x += m->tile_size) {
        u32 copies = m->frequency[k];
        for (i64 i = 0; i < m->tile_size; i++) {
            counts[x[i]] += copies;
        }
    }
    i64 n = MEASURED_TILES * m->tile_size;
    f64 sum = 0, squares = 0;
    for (int v = 0; v < 65536; v++) {
        if (counts[v]) {
            u16 bits = v;
            f16 value;
            memcpy(&value, &bits, 2);
            sum += (f64)counts[v] * (f64)value;
        }
    }
    f64 mean = sum / n;
    for (int v = 0; v < 65536; v++) {
        if (counts[v]) {
            u16 bits = v;
            f16 value;
            memcpy(&value, &bits, 2);
            f64 d = (f64)value - mean;
            squares += (f64)counts[v] * (d * d);
        }
    }
    m->mean[c] = mean;
    m->variance[c] = squares / n;
}

void run_grid(Reader *r, id<MTLComputePipelineState> p, NSArray<id<MTLBuffer>> *buffers,
    const void *bytes, int size, MTLSize grid) {
    id<MTLCommandBuffer> command = [r->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:p];
    for (NSUInteger i = 0; i < buffers.count; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [encoder setBytes:bytes length:size atIndex:buffers.count];
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
}

typedef struct {
    f16 table[256], w1[27 * STEM], b1[STEM], w2[27 * STEM * STEM], b2[STEM];
} Stats;

static void measure_window(Reader *r, const u8 *stack, int h, int w, Stats *out) {
    pthread_mutex_lock(&r->measuring);
    int sy, sx, tiles = 0, frequency[MEASURED_TILES] = {0};
    i64 corners[MEASURED_TILES][2], unique[MEASURED_TILES][2];
    measured_tiles(stack, DEPTH, h, w, corners, &sy, &sx);
    for (int i = 0; i < MEASURED_TILES; i++) {
        int k = 0;
        while (k < tiles && (unique[k][0] != corners[i][0] || unique[k][1] != corners[i][1])) {
            k++;
        }
        if (k == tiles) {
            unique[tiles][0] = corners[i][0];
            unique[tiles++][1] = corners[i][1];
        }
        frequency[k]++;
    }
    pack_measurement(stack, h, w, sy, sx, unique, tiles, r->tiles.contents);
    Scale s = raw_scale(stack, DEPTH, h, w, corners, sy, sx);
    f16 *table = r->measure_table.contents;
    for (int v = 0; v < 256; v++) {
        f16 x = v, low = s.low, high = s.high;
        f32 clamped = x < low ? low : x > high ? high : x;
        table[v] = (f32)(f16)(clamped - (f32)s.centre) / (f32)s.spread;
        out->table[v] = table[v];
    }
    i64 tile_size = (i64)DEPTH * sy * sx;
    Box box = {tiles, sy, sx, tiles * tile_size};
    MTLSize grid = MTLSizeMake(sx, sy, tiles * DEPTH);
    run_grid(r, r->first, @[ r->tiles, r->measure_table, r->conv1, r->measured1 ], &box,
        sizeof(box), grid);
    Stem *k = &r->stem_weights;
    f32 scale1[STEM], shift1[STEM], scale2[STEM], shift2[STEM], mean[STEM], variance[STEM];
    Moments m = {r->measured1.contents, tile_size, tiles, frequency, r->bins, mean, variance};
    parallel(STEM, STEM, channel_moments_of, &m);
    scale_shift(k->norms[0], k->norms[1], mean, variance, scale1, shift1);
    f16 *norm1 = r->norm1.contents;
    for (int c = 0; c < STEM; c++) {
        norm1[c] = scale1[c];
        norm1[STEM + c] = shift1[c];
    }
    id<MTLCommandBuffer> command = [r->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:r->second];
    [encoder setBuffer:r->measured1 offset:0 atIndex:0];
    [encoder setBuffer:r->conv2 offset:0 atIndex:1];
    [encoder setBuffer:r->norm1 offset:0 atIndex:2];
    [encoder setBuffer:r->measured2 offset:0 atIndex:3];
    [encoder setBytes:&box length:sizeof(box) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((sx + 15) / 16, (sy + 7) / 8, tiles * DEPTH)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    m.x = r->measured2.contents;
    parallel(STEM, STEM, channel_moments_of, &m);
    scale_shift(k->norms[2], k->norms[3], mean, variance, scale2, shift2);
    pthread_mutex_unlock(&r->measuring);
    Folded f = fold(k, scale1, shift1, scale2, shift2);
    for (int i = 0; i < 27 * STEM * STEM; i++) {
        out->w2[i] = f.w2[i];
        out->w1[i % (27 * STEM)] = f.w1[i % (27 * STEM)];
    }
    for (int c = 0; c < STEM; c++) {
        out->b1[c] = f.b1[c];
        out->b2[c] = f.b2[c];
    }
}

void measure_stack(Reader *r, const u8 *stack, int h, int w, Stats *out) {
    @autoreleasepool {
        measure_window(r, stack, h, w, out);
    }
}

void use_stats(Reader *r, const Stats *s) {
    memcpy(r->table.contents, s->table, sizeof(s->table));
    memcpy(r->w1.contents, s->w1, sizeof(s->w1));
    memcpy(r->b1.contents, s->b1, sizeof(s->b1));
    memcpy(r->w2.contents, s->w2, sizeof(s->w2));
    memcpy(r->b2.contents, s->b2, sizeof(s->b2));
}

void fix_stem(Reader *r, const u8 *stack, int h, int w) {
    Stats s;
    measure_stack(r, stack, h, w, &s);
    use_stats(r, &s);
}

void stem_tile(Reader *r, int slot, int ty, int tx) {
    StemDims dims = {ty, tx, r->stem_weights.attention_bias};
    id<MTLCommandBuffer> command = [r->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:r->stem];
    id<MTLBuffer> buffers[] = {
        r->block[slot], r->w1, r->b1, r->w2, r->b2, r->att, r->features[slot]};
    for (int i = 0; i < 7; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [encoder setBytes:&dims length:sizeof(dims) atIndex:7];
    [encoder setBuffer:r->table offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(tx / 16, ty / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
}

void engine_tile(Reader *r, int slot) {
    NSError *error = nil;
    MLMultiArray *input = [[MLMultiArray alloc]
        initWithDataPointer:r->features[slot].contents
                      shape:@[ @1, @(2 * STEM), @TILE, @TILE ]
                   dataType:MLMultiArrayDataTypeFloat16
                    strides:@[ @(2 * STEM * TILE * TILE), @(TILE * TILE), @TILE, @1 ]
                deallocator:nil
                      error:&error];
    MLDictionaryFeatureProvider *given =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"features" : input}
                                                          error:&error];
    f32 *read = r->reads[slot];
    const f32 *sigmoid = r->sigmoid;
    int *failed = r->failed + slot;
    dispatch_semaphore_t done = r->done[slot];
    [r->engine predictionFromFeatures:given
                    completionHandler:^(id<MLFeatureProvider> output, NSError *failure) {
                      MLMultiArray *ink = [output featureValueForName:@"ink"].multiArrayValue;
                      *failed = !ink || ink.dataType != MLMultiArrayDataTypeFloat16 ||
                          ink.strides.lastObject.integerValue != 1;
                      if (!*failed) {
                          [ink getBytesWithHandler:^(const void *bytes, NSInteger size) {
                            i64 pitch = ink.strides[ink.strides.count - 2].integerValue;
                            for (int y = 0; y < TILE; y++) {
                                const u16 *row = (const u16 *)bytes + y * pitch;
                                for (int x = 0; x < TILE; x++) {
                                    read[y * TILE + x] = sigmoid[row[x]];
                                }
                            }
                          }];
                      }
                      dispatch_semaphore_signal(done);
                    }];
}

void finish_flows(Reader *r) {
    while (r->finished < r->queued) {
        Flow *f = r->flows + r->finished % FLOWS;
        if (f->settled < f->count) {
            return;
        }
        blend_weights(f->blend, f->places, f->count, f->tx, r->weight, f->h, f->w);
        blend_finish(f->total, r->weight, (i64)f->h * f->w);
        r->finished++;
    }
}

void settle_oldest(Reader *r) {
    int slot = r->settled % SLOTS;
    dispatch_semaphore_wait(r->done[slot], DISPATCH_TIME_FOREVER);
    need(!r->failed[slot], "engine call failed");
    Flow *f = r->flows + r->slot_flow[slot] % FLOWS;
    blend_add(f->blend, r->slot_place[slot], f->tx, r->reads[slot], f->total, f->w);
    f->settled++;
    r->settled++;
    finish_flows(r);
}

int ink_ready(const Reader *r, i64 id) { return id < r->finished; }

void ink_finish_through(Reader *r, i64 id) {
    while (r->finished <= id) {
        need(r->settled < r->issued, "a read was never queued");
        settle_oldest(r);
    }
}

i64 ink_queue(Reader *r, const u8 *stack, int h, int w, f32 *total) {
    while (r->finished <= r->queued - FLOWS) {
        settle_oldest(r);
    }
    Flow *f = r->flows + r->queued % FLOWS;
    f->count = plan_tiles(&r->work, stack, h, w, &f->ty, &f->tx);
    f->total = total;
    f->h = h;
    f->w = w;
    f->settled = 0;
    memcpy(f->places, r->work.places, f->count * sizeof(Place));
    memcpy(f->blend, r->work.blend, (i64)f->ty * f->tx * sizeof(f32));
    memset(total, 0, (i64)h * w * sizeof(f32));
    for (int k = 0; k < f->count; k++) {
        while (r->issued - r->settled >= SLOTS) {
            settle_oldest(r);
        }
        int slot = r->issued % SLOTS;
        @autoreleasepool {
            cut_tile(&r->work, f->places[k], h, w, f->ty, f->tx, r->block[slot].contents);
            stem_tile(r, slot, f->ty, f->tx);
            r->slot_flow[slot] = r->queued;
            r->slot_place[slot] = f->places[k];
            if (f->ty == TILE && f->tx == TILE && r->engine) {
                engine_tile(r, slot);
            } else {
                body_tile(r, slot, f->ty, f->tx);
            }
        }
        r->issued++;
    }
    i64 id = r->queued++;
    finish_flows(r);
    return id;
}

void ink_read(Reader *r, const u8 *stack, int h, int w, f32 *total) {
    ink_finish_through(r, ink_queue(r, stack, h, w, total));
}
