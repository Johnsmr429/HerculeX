#import <CoreML/CoreML.h>
#include <string.h>

// Call Core ML without holding the Python GIL.

static MLModel *opened(NSURL *compiled, int units, int specialized) {
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = (MLComputeUnits)units;
    if (specialized) {
        MLOptimizationHints *hints = [MLOptimizationHints new];
        hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
        configuration.optimizationHints = hints;
    }
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
    if (!model) NSLog(@"engine: %@", error);
    return model;
}

void *hx_engine_open(const char *package, const char *compiled_path, int units, int specialized) {
    @autoreleasepool {
        NSFileManager *files = [NSFileManager defaultManager];
        NSURL *compiled = [NSURL fileURLWithPath:[NSString stringWithUTF8String:compiled_path]];
        if (![files fileExistsAtPath:compiled.path]) {
            NSError *error = nil;
            NSURL *made = [MLModel compileModelAtURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:package]] error:&error];
            if (!made) {
                NSLog(@"engine: %@", error);
                return NULL;
            }
            if (![files moveItemAtURL:made toURL:compiled error:&error]) [files removeItemAtURL:made error:nil];
        }
        MLModel *model = opened(compiled, units, specialized);
        return model ? (__bridge_retained void *)model : NULL;
    }
}

// One prediction from the form safe from two threads at once, its logits copied out as [side, side] half precision.
static int answered(MLModel *model, id<MLFeatureProvider> given, int side, void *logits) {
    __block id<MLFeatureProvider> found = nil;
    dispatch_semaphore_t answer = dispatch_semaphore_create(0);
    [model predictionFromFeatures:given completionHandler:^(id<MLFeatureProvider> output, NSError *failure) {
        if (!output) NSLog(@"engine: %@", failure);
        found = output;
        dispatch_semaphore_signal(answer);
    }];
    dispatch_semaphore_wait(answer, DISPATCH_TIME_FOREVER);
    if (!found) return 3;
    MLMultiArray *ink = [found featureValueForName:@"ink"].multiArrayValue;
    if (ink.dataType != MLMultiArrayDataTypeFloat16 || ink.count != side * side) return 4;
    __block int failed = 0;
    [ink getBytesWithHandler:^(const void *bytes, NSInteger size) {
        NSInteger rowStride = ink.strides[ink.strides.count - 2].integerValue;
        for (int y = 0; y < side; y++) memcpy((uint16_t *)logits + (size_t)y * side, (const uint16_t *)bytes + (size_t)y * rowStride, (size_t)side * sizeof(uint16_t));
        failed = ink.strides.lastObject.integerValue != 1;
    }];
    return failed ? 5 : 0;
}

static MLMultiArray *half_array(void *data, int channels, int side) {
    NSError *error = nil;
    NSNumber *plane = @(side * side), *row = @(side), *depth = @(channels);
    return [[MLMultiArray alloc] initWithDataPointer:data shape:@[@1, depth, row, row] dataType:MLMultiArrayDataTypeFloat16 strides:@[@(channels * side * side), plane, row, @1] deallocator:nil error:&error];
}

// Logits of one tile: features [1, channels, side, side] and logits [side, side], both half precision.
int hx_engine_read(void *engine, void *features, int channels, int side, void *logits) {
    @autoreleasepool {
        NSError *error = nil;
        MLMultiArray *input = half_array(features, channels, side);
        if (!input) return 1;
        MLDictionaryFeatureProvider *given = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"features": input} error:&error];
        if (!given) return 2;
        return answered((__bridge MLModel *)engine, given, side, logits);
    }
}

// Logits of one tile of the whole network: image [1, layers, side, side] and four norms of channels each, all half precision.
int hx_engine_whole(void *engine, void *image, int layers, void *norms, int channels, int side, void *logits) {
    @autoreleasepool {
        NSError *error = nil;
        NSMutableDictionary *inputs = [NSMutableDictionary dictionaryWithObject:half_array(image, layers, side) forKey:@"image"];
        NSArray *names = @[@"scale_1", @"shift_1", @"scale_2", @"shift_2"];
        for (int k = 0; k < 4; k++) inputs[names[k]] = half_array((uint16_t *)norms + (size_t)k * channels, channels, 1);
        MLDictionaryFeatureProvider *given = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&error];
        if (!given) return 2;
        return answered((__bridge MLModel *)engine, given, side, logits);
    }
}
