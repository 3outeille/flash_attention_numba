#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\nDevice #%d:\n", i);
        printf("  Device name: %s\n", deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        // reservedSharedMemPerBlock
        printf("  Shared memory reserved by CUDA driver per block in bytes: %lu bytes\n", deviceProp.reservedSharedMemPerBlock);
        // sharedMemPerBlock
        printf("  Shared memory available per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
        // sharedMemPerMultiprocessor
        printf("  Shared memory available per multiprocessor: %lu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        // 96 KB = 96 * 1024 = 98304 bytes
        printf("  Shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    }

    return 0;
}