#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdint.h>
#include <string.h>

void need(int ok, const char *format, ...);

@interface SliceMetalPlan : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input, *output;
@end

@implementation SliceMetalPlan
@end

@interface SliceMetalModel : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLBuffer> input;
@property(nonatomic, strong) NSMutableDictionary<NSValue *, SliceMetalPlan *> *plans;
@end

@implementation SliceMetalModel
@end

int slices_accelerator_available(void) { return MTLCreateSystemDefaultDevice() != nil; }

void *slices_accelerator_open(uint64_t seed) {
    SliceMetalModel *metal = [SliceMetalModel new];
    metal.device = MTLCreateSystemDefaultDevice();
    metal.queue = [metal.device newCommandQueue];
    need(metal.queue != nil, "Synthetic-Slices needs an Apple Metal device");
    metal.plans = [NSMutableDictionary new];
    return (__bridge_retained void *)metal;
}

void slices_accelerator_close(void *ptr) { CFRelease(ptr); }

static SliceMetalPlan *slices_metal_plan(SliceMetalModel *model, const float *weights, int ci,
    int co, int side, int kernel, int stride) {
    NSValue *key = [NSValue valueWithPointer:weights];
    SliceMetalPlan *plan = model.plans[key];
    if (plan) {
        return plan;
    }
    plan = [SliceMetalPlan new];
    MPSGraph *graph = [MPSGraph new];
    plan.graph = graph;
    plan.input = [graph placeholderWithShape:@[ @1, @(ci), @(side), @(side) ]
                                    dataType:MPSDataTypeFloat32
                                        name:nil];
    NSData *data = [NSData dataWithBytes:weights length:(size_t)ci * co * kernel * kernel * 4];
    MPSGraphTensor *weight = [graph constantWithData:data
                                               shape:@[ @(co), @(ci), @(kernel), @(kernel) ]
                                            dataType:MPSDataTypeFloat32];
    int pad = kernel / 2;
    MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
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
    plan.output = [graph convolution2DWithSourceTensor:plan.input
                                         weightsTensor:weight
                                            descriptor:desc
                                                  name:nil];
    model.plans[key] = plan;
    return plan;
}

void slices_accelerator_conv(void *ptr, const float *weights, const float *input, float *out,
    int ci, int co, int side, int kernel, int stride) {
    @autoreleasepool {
        SliceMetalModel *model = (__bridge SliceMetalModel *)ptr;
        SliceMetalPlan *plan = slices_metal_plan(model, weights, ci, co, side, kernel, stride);
        size_t bytes = (size_t)ci * side * side * sizeof(float);
        if (model.input.length < bytes) {
            model.input = [model.device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
            need(model.input != nil, "cannot allocate Synthetic-Slices Metal input");
        }
        memcpy(model.input.contents, input, bytes);
        MPSGraphTensorData *tensor =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:model.input
                                                    shape:@[ @1, @(ci), @(side), @(side) ]
                                                 dataType:MPSDataTypeFloat32];
        NSDictionary *result = [plan.graph runWithMTLCommandQueue:model.queue
                                                            feeds:@{plan.input : tensor}
                                                    targetTensors:@[ plan.output ]
                                                 targetOperations:nil];
        MPSGraphTensorData *y = result[plan.output];
        need(y != nil, "Synthetic-Slices Metal convolution failed");
        [y.mpsndarray readBytes:out strideBytes:nil];
    }
}
