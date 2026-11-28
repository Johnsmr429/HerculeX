static int finish_render(id<MTLCommandBuffer> command, id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline, id<MTLBuffer> result, MTLSize grid, uint8_t *out,
    size_t size, double *seconds) {
    NSUInteger width = pipeline.threadExecutionWidth;
    NSUInteger height = MIN(pipeline.maxTotalThreadsPerThreadgroup / width, 8);
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.error)
        return 3;
    *seconds = command.GPUEndTime - command.GPUStartTime;
    memcpy(out, result.contents, size);
    return 0;
}
