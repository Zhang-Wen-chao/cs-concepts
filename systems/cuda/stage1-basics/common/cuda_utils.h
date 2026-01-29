#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA 错误检查宏
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// CPU 计时器
inline double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// 初始化数组
void initialData(float *data, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void initialData_int(int *data, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        data[i] = (int)(rand() & 0xFF);
    }
}

// 验证结果
void checkResult(float *hostRef, float *gpuRef, const int N, float epsilon = 1.0e-4) {
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

// 打印设备信息
void printDeviceInfo() {
    int devID = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    printf("Device %d: %s\n", devID, deviceProp.name);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total Global Memory: %.2f GB\n",
           (float)deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Warp Size: %d\n", deviceProp.warpSize);
    printf("\n");
}

// CUDA 事件计时器封装
class CudaTimer {
private:
    cudaEvent_t start, stop;

public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start, 0);
    }

    float stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

#endif // CUDA_UTILS_H
