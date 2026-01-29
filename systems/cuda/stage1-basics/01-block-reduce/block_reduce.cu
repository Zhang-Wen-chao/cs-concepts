#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/cuda_utils.h"

// CPU 端的规约实现（用于验证）
int reduceOnCPU(int *data, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

// TODO: 实现 Neighbored Pairing 规约
// 每个线程处理相邻的元素
// 步长从 1 开始，每次翻倍
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    // 获取线程 ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx >= n) return;

    // 将全局数据指针指向当前 block 的数据
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // TODO: 实现规约逻辑
    // 提示：使用 for 循环，stride 从 1 开始每次 *2
    // 提示：需要使用 __syncthreads() 同步
    // 提示：只有 tid % (2*stride) == 0 的线程参与计算

    // 将结果写回全局内存（只有线程 0 写）
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

// TODO: 实现 Interleaved Pairing 规约
// 交错访问模式，性能更好
// 步长从 blockDim.x/2 开始，每次减半
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    // TODO: 实现交错规约逻辑
    // 提示：stride 从 blockDim.x/2 开始，每次 >>= 1（右移1位，相当于除以2）
    // 提示：条件是 tid < stride
    // 提示：访问 idata[tid] 和 idata[tid + stride]

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

int main(int argc, char **argv) {
    // 打印设备信息
    printDeviceInfo();

    // 设置数组大小
    int size = 1 << 24;  // 16M 个元素
    printf("Array size: %d\n", size);

    // 设置 block size
    int blockSize = 256;
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }

    // 计算 grid size
    dim3 block(blockSize);
    dim3 grid((size + block.x - 1) / block.x);
    printf("Grid size: %d, Block size: %d\n\n", grid.x, block.x);

    // 分配 host 内存
    size_t bytes = size * sizeof(int);
    int *h_idata = (int*)malloc(bytes);
    int *h_odata = (int*)malloc(grid.x * sizeof(int));

    // 初始化数据
    initialData_int(h_idata, size);

    // 分配 device 内存
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));

    // ========== CPU 规约 ==========
    double iStart = cpuSecond();
    int cpu_sum = reduceOnCPU(h_idata, size);
    double iElaps = cpuSecond() - iStart;
    printf("CPU reduce:           %d, elapsed %.6f sec\n", cpu_sum, iElaps);

    // ========== GPU Neighbored 规约 ==========
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.startTimer();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    float gpu_time = timer.stopTimer();

    // 将每个 block 的结果拷贝回 host 并求和
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_odata[i];
    }

    printf("GPU Neighbored:       %d, elapsed %.6f ms", gpu_sum, gpu_time);
    if (gpu_sum == cpu_sum) {
        printf(" ✓\n");
    } else {
        printf(" ✗ (MISMATCH!)\n");
    }

    // ========== GPU Interleaved 规约 ==========
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    timer.startTimer();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    gpu_time = timer.stopTimer();

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_odata[i];
    }

    printf("GPU Interleaved:      %d, elapsed %.6f ms", gpu_sum, gpu_time);
    if (gpu_sum == cpu_sum) {
        printf(" ✓\n");
    } else {
        printf(" ✗ (MISMATCH!)\n");
    }

    // 释放内存
    free(h_idata);
    free(h_odata);
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // Reset device
    CHECK(cudaDeviceReset());

    return 0;
}
