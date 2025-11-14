/*
 * Exercise 10: CUDA Vector Addition
 * 
 * Objective: Write your first CUDA kernel for vector addition.
 * 
 * Tasks:
 * 1. Implement vector addition kernel
 * 2. Launch kernel with appropriate grid/block configuration
 * 3. Verify correctness
 * 4. Profile performance
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// TODO: Implement vector addition kernel
// __global__ void vectorAdd(float *A, float *B, float *C, int N)
// {
//     // Calculate thread index
//     // Hint: Use blockIdx, blockDim, threadIdx
//     // int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     
//     // Check bounds and perform addition
//     // if (idx < N) {
//     //     C[idx] = A[idx] + B[idx];
//     // }
// }

// CPU reference implementation
void vectorAddCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    printf("============================================================\n");
    printf("Exercise 10: CUDA Vector Addition\n");
    printf("============================================================\n");
    
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // TODO: Launch kernel
    // Configure grid and block dimensions
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    vectorAddCPU(h_A, h_B, h_C_ref, N);
    
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Error at index %d: %f != %f\n", i, h_C[i], h_C_ref[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Result: PASSED\n");
    } else {
        printf("Result: FAILED\n");
    }
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("============================================================\n");
    printf("Exercise complete! Implement the TODO above.\n");
    printf("============================================================\n");
    
    return 0;
}

