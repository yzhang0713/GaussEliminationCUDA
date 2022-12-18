
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <algorithm>

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

constexpr auto DEBUG = 0;

void findMaxMagnitude(double** A, double* s, int size);

void pivot(double** A, double* b, double* s, int size, int index);

int eliminate(double** A, double* b, double* s, int size, double tol);

void substitute(double** A, double* b, double* x, int size);

int gaussElimination(double** A, double* b, double* x, int size, double tol);

void initialData(double* v, int size);

void checkResult(double* hostRef, double* gpuRef, int size);

void printVector(double* v, int size) {
    std::cout << "\t";
    for (int i = 0; i < size; i++) {
        std::cout << v[i] << "\t";
    }
    std::cout << std::endl;
}

int checkSolution(double** A, double* b, double* x, int size, double tol);

void copyArray(double* a, double* b, int size);

__global__ void reduceForMax(double* g_idata, double* g_odata, unsigned int size) {
    // Get thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get local data pointer
    double* local_data = g_idata + blockIdx.x * blockDim.x;

    // Boundary check
    if (idx >= size) {
        return;
    }

    // In place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_data[tid] = fmax(fabs(local_data[tid]), fabs(local_data[tid + stride]));
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = local_data[0];
    }
}

void findMaxMagnitudeGPU(double** A, double* s, int size);

__global__ void swapVectors(double* v1, double* v2, unsigned int size) {
    // Get thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx >= size) {
        return;
    }

    // Swap the current element
    double temp = v1[idx];
    v1[idx] = v2[idx];
    v2[idx] = temp;
}

__global__ void eliminateOnGPU(double* v1, double* v2, double factor, unsigned int size) {
    // Get thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx >= size) {
        return;
    }

    // Swap the current element
    v2[idx] -= factor * v1[idx];
}

void swapRowsOnGPU(double** A, unsigned int row1, unsigned int row2, unsigned int start, unsigned int size, dim3 block, dim3 grid);

void eliminateRowsOnGPU(double** A, unsigned int row1, unsigned int row2, double factor, unsigned int start, unsigned int size, dim3 block, dim3 grid);

void pivotGPU(double** A, double* b, double* s, int size, int index, dim3 block, dim3 grid);

int eliminateGPU(double** A, double* b, double* s, int size, double tol, dim3 block, dim3 grid);

int gaussEliminationGPU(double** A, double* b, double* x, int size, double tol);

cudaError_t addWithCuda(int *c, const int *a, const int *b, int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{
    // Seed the random generator
    time_t t;
    srand((unsigned int)time(&t));

    // Check command line arguments
    if (argc != 2) {
        printf("The number of arguments are incorrect, please provide a second argument as the size of matrix.\n");
        return 0;
    }

    // Get the size of problem
    int size = atoi(argv[1]);

    // Define the value of tolerant
    double tol = 1.0E-8;

    // Values to hold time
    clock_t t_start, t_diff;
    int msec;

    // Allocate and populate matrix A, vector b randomly
    double** A = (double**)malloc(sizeof(double*) * size);
    for (int i = 0; i < size; i++) {
        A[i] = (double*)malloc(sizeof(double) * size);
        initialData(A[i], size);
    }
    double* b = (double*)malloc(sizeof(double) * size);
    initialData(b, size);

    if (DEBUG) {
        std::cout << "A matrix:" << std::endl;
        for (int i = 0; i < size; i++) {
            printVector(A[i], size);
        }
        std::cout << "b vector:" << std::endl;
        printVector(b, size);
    }

    // Copy the matrix to the ones used for host side and device side
    double** A_host = (double**)malloc(sizeof(double*) * size);
    for (int i = 0; i < size; i++) {
        A_host[i] = (double*)malloc(sizeof(double) * size);
        copyArray(A[i], A_host[i], size);
    }
    double* b_host = (double*)malloc(sizeof(double) * size);
    copyArray(b, b_host, size);

    double** A_device = (double**)malloc(sizeof(double*) * size);
    for (int i = 0; i < size; i++) {
        A_device[i] = (double*)malloc(sizeof(double) * size);
        copyArray(A[i], A_device[i], size);
    }
    double* b_device = (double*)malloc(sizeof(double) * size);
    copyArray(b, b_device, size);

    // Allocate vector x_host, x_device
    double* x_host = (double*)malloc(sizeof(double) * size);
    memset(x_host, 0, sizeof(double) * size);
    double* x_device = (double*)malloc(sizeof(double) * size);
    memset(x_device, 0, sizeof(double) * size);

    t_start = clock();
    int error_flag_host = gaussElimination(A_host, b_host, x_host, size, tol);
    t_diff = clock() - t_start;

    msec = t_diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken to process Gauss Elimination on CPU: %d seconds %d milliseconds.\n", msec / 1000, msec % 1000);

    if (DEBUG) {
        std::cout << "x_host vector: " << std::endl;
        printVector(x_host, size);
    }

    if (DEBUG) {
        int result = checkSolution(A, b, x_host, size, tol);
        if (result) {
            printf("Host result matches.\n");
        }
        else {
            printf("Host result does not match.\n");
        }
    }

    if (error_flag_host == -1) {
        printf("Failed to get solution of Ax=b from host side.\n");
        exit(1);
    }

    t_start = clock();
    int error_flag_device = gaussEliminationGPU(A_device, b_device, x_device, size, tol);
    t_diff = clock() - t_start;

    msec = t_diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken to process Gauss Elimination on GPU: %d seconds %d milliseconds.\n", msec / 1000, msec % 1000);

    if (DEBUG) {
        std::cout << "x_device vector: " << std::endl;
        printVector(x_device, size);
    }

    if (DEBUG) {
        int result = checkSolution(A, b, x_device, size, tol);
        if (result) {
            printf("Device result matches.\n");
        }
        else {
            printf("Device result does not match.\n");
        }
    }

    if (error_flag_device == -1) {
        printf("Failed to get solution of Ax=b from device side.\n");
        exit(1);
    }


    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    free(b);
    for (int i = 0; i < size; i++) {
        free(A[i]);
    }
    free(A);
    free(b_host);
    for (int i = 0; i < size; i++) {
        free(A_host[i]);
    }
    free(A_host);
    free(b_device);
    for (int i = 0; i < size; i++) {
        free(A_device[i]);
    }
    free(A_device);
    free(x_host);
    free(x_device);

    return 0;
}

// This function will find the maximum magnitude of each row in matrix A, and put the value in vector s
//    Matrix A: size X size
//    Vector s: size X 1
void findMaxMagnitude(double** A, double* s, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        s[i] = fabs(A[i][0]);
        for (j = 1; j < size; j++) {
            if (abs(A[i][j] > s[i])) {
                s[i] = fabs(A[i][j]);
            }
        }
    }
}

// This function will do the pivoting step of Gauss Elimination
//    Matrix A: size X size
//    Vector b: size X 1
//    Vector s: size X 1
void pivot(double** A, double* b, double* s, int size, int index) {
    double current_max = fabs(A[index][index]) / s[index];
    unsigned int current_ind = index;
    int i;
    double dummy;
    for (i = index + 1; i < size; i++) {
        dummy = fabs(A[i][i]) / s[i];
        if (dummy > current_max) {
            current_max = dummy;
            current_ind = i;
        }
    }
    if (current_ind != index) {
        for (i = index; i < size; i++) {
            dummy = A[current_ind][i];
            A[current_ind][i] = A[index][i];
            A[index][i] = dummy;
        }
        dummy = b[current_ind];
        b[current_ind] = b[index];
        b[index] = dummy;
        dummy = s[current_ind];
        s[current_ind] = s[index];
        s[index] = dummy;
    }
}


// This function will do the elimination step of Gauss Elimination
//    Matrix A: size X size
//    Vector b: size X 1
//    Vector s: size X 1
int eliminate(double** A, double* b, double* s, int size, double tol) {
    int error_flag = 0;
    int index, i, j;
    for (index = 0; index < size-1; index++) {
        // For the elimination with regards to each col, first do the pivoting of that row to ensure most stable elimination step can be done
        pivot(A, b, s, size, index);
        if (fabs(A[index][index]) / s[index] < tol) {
            // Encounter zero or very small elimination element, return error state
            error_flag = -1;
            break;
        }
        for (i = index + 1; i < size; i++) {
            double factor = A[i][index] / A[index][index];
            for (j = index + 1; j < size; j++) {
                A[i][j] -= factor * A[index][j];
            }
            b[i] -= factor * b[index];
        }
    }
    if (fabs(A[size - 1][size - 1]) / s[size - 1] < tol) {
        error_flag = -1;
    }
    return error_flag;
}


// This function will do the substitution step of Gauss Elimination
//    Matrix A: size X size
//    Vector b: size X 1
//    Vector x: size X 1
void substitute(double** A, double* b, double* x, int size) {
    x[size - 1] = b[size - 1] / A[size - 1][size - 1];
    int i, j;
    double sum;
    for (i = size - 2; i >= 0; i--) {
        sum = 0;
        for (j = i + 1; j < size; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
}


// This function will do Gauss Elimination: Ax = b
//    Matrix A: size X size
//    Vector b: size X 1
//    Vector x: size X 1
int gaussElimination(double** A, double* b, double* x, int size, double tol) {
    double* s = (double*)malloc(sizeof(double) * size);
    int error_flag = 0;
    findMaxMagnitude(A, s, size);
    if (DEBUG) {
        std::cout << "s vector: " << std::endl;
        printVector(s, size);
    }
    error_flag = eliminate(A, b, s, size, tol);
    if (DEBUG) {
        std::cout << "A matrix: " << std::endl;
        for (int i = 0; i < size; i++) {
            printVector(A[i], size);
        }
    }
    if (error_flag != -1) {
        substitute(A, b, x, size);
    }
    free(s);
    return error_flag;
}

void findMaxMagnitudeGPU(double** A, double* s, int size, dim3 block, dim3 grid) {
    // Number of bytes for each row
    size_t bytes = sizeof(double) * size;

    // Allocate device memory
    double* d_idata = NULL;
    double* d_odata = NULL;
    CHECK(cudaMalloc((void**)&d_idata, bytes));
    CHECK(cudaMalloc((void**)&d_odata, sizeof(double) * grid.x));

    // Allocate host memory for temporary results
    double* temp = (double*)malloc(sizeof(double) * grid.x);

    // For each row, do reduction to find max
    for (int i = 0; i < size; i++) {
        memset(temp, 0, sizeof(double) * grid.x);
        CHECK(cudaMemcpy(d_idata, A[i], bytes, cudaMemcpyHostToDevice));
        CHECK(cudaDeviceSynchronize());
        reduceForMax <<<grid, block >>> (d_idata, d_odata, size);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(temp, d_odata, sizeof(double) * grid.x, cudaMemcpyDeviceToHost));
        s[i] = temp[0];
        for (int j = 1; j < grid.x; j++) {
            s[i] = std::max(s[i], temp[j]);
        }
    }

    // Free resource
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));
    free(temp);
}

void swapRowsOnGPU(double** A, unsigned int row1, unsigned int row2, unsigned int start, unsigned int size, dim3 block, dim3 grid) {

    // Number of bytes for each row
    size_t bytes = sizeof(double) * (size - start);

    // Allocate device memory
    double* d_row1 = NULL;
    double* d_row2 = NULL;
    CHECK(cudaMalloc((void**)&d_row1, bytes));
    CHECK(cudaMalloc((void**)&d_row2, bytes));

    // For each row, do reduction to find max
    CHECK(cudaMemcpy(d_row1, A[row1] + start, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_row2, A[row2] + start, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    swapVectors <<<grid, block >>> (d_row1, d_row2, size-start);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(A[row1] + start, d_row1, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(A[row2] + start, d_row2, bytes, cudaMemcpyDeviceToHost));

    // Free resource
    CHECK(cudaFree(d_row1));
    CHECK(cudaFree(d_row2));

}

void eliminateRowsOnGPU(double** A, unsigned int row1, unsigned int row2, double factor, unsigned int start, unsigned int size, dim3 block, dim3 grid) {

    // Number of bytes for each row
    size_t bytes = sizeof(double) * (size - start);

    // Allocate device memory
    double* d_row1 = NULL;
    double* d_row2 = NULL;
    CHECK(cudaMalloc((void**)&d_row1, bytes));
    CHECK(cudaMalloc((void**)&d_row2, bytes));

    // For each row, do reduction to find max
    CHECK(cudaMemcpy(d_row1, A[row1] + start, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_row2, A[row2] + start, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    eliminateOnGPU <<<grid, block >>> (d_row1, d_row2, factor, size - start);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(A[row2] + start, d_row2, bytes, cudaMemcpyDeviceToHost));

    // Free resource
    CHECK(cudaFree(d_row1));
    CHECK(cudaFree(d_row2));

}

void pivotGPU(double** A, double* b, double* s, int size, int index, dim3 block, dim3 grid) {
    double current_max = fabs(A[index][index]) / s[index];
    unsigned int current_ind = index;
    int i;
    double dummy;
    for (i = index + 1; i < size; i++) {
        dummy = fabs(A[i][i]) / s[i];
        if (dummy > current_max) {
            current_max = dummy;
            current_ind = i;
        }
    }
    if (current_ind != index) {
        swapRowsOnGPU(A, index, current_ind, index, size, block, grid);
        dummy = b[current_ind];
        b[current_ind] = b[index];
        b[index] = dummy;
        dummy = s[current_ind];
        s[current_ind] = s[index];
        s[index] = dummy;
    }
}

int eliminateGPU(double** A, double* b, double* s, int size, double tol, dim3 block, dim3 grid) {
    int error_flag = 0;
    int index, i, j;
    for (index = 0; index < size - 1; index++) {
        // For the elimination with regards to each col, first do the pivoting of that row to ensure most stable elimination step can be done
        pivotGPU(A, b, s, size, index, block, grid);
        if (fabs(A[index][index]) / s[index] < tol) {
            // Encounter zero or very small elimination element, return error state
            error_flag = -1;
            break;
        }
        for (i = index + 1; i < size; i++) {
            double factor = A[i][index] / A[index][index];
            //eliminateRowsOnGPU(A, index, i, factor, index, size, block, grid);
            for (j = index + 1; j < size; j++) {
                A[i][j] -= factor * A[index][j];
            }
            b[i] -= factor * b[index];
        }
    }
    if (fabs(A[size - 1][size - 1]) / s[size - 1] < tol) {
        error_flag = -1;
    }
    return error_flag;
}

int gaussEliminationGPU(double** A, double* b, double* x, int size, double tol) {
    //// Allocate device memory and copy values from host to device
    //double** A_device = NULL;
    //double* b_device = NULL;
    //double* x_device = NULL;
    //CHECK(cudaMalloc((void***)&A_device, sizeof(double*) * size));
    //for (int i = 0; i < size; i++) {
    //    CHECK(cudaMalloc((void**)&A_device[i], sizeof(double) * size));
    //    CHECK(cudaMemcpy(A_device[i], A[i], sizeof(double) * size, cudaMemcpyHostToDevice));
    //}
    //CHECK(cudaMalloc((void**)&b_device, sizeof(double) * size));
    //CHECK(cudaMemcpy(b_device, b, sizeof(double) * size, cudaMemcpyHostToDevice));
    //CHECK(cudaMalloc((void**)&x_device, sizeof(double) * size));
    //CHECK(cudaDeviceSynchronize());

    double* s = (double*)malloc(sizeof(double) * size);

   /* double* s = NULL;
    CHECK(cudaMalloc((void**)&s, sizeof(double) * size));*/

    // Define CUDA dimensions
    dim3 block(512, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    int error_flag = 0;
    findMaxMagnitudeGPU(A, s, size, block, grid);
    if (DEBUG) {
        std::cout << "s vector: " << std::endl;
        printVector(s, size);
    }
    error_flag = eliminateGPU(A, b, s, size, tol, block, grid);
    if (DEBUG) {
        std::cout << "A matrix: " << std::endl;
        for (int i = 0; i < size; i++) {
            printVector(A[i], size);
        }
    }
    if (error_flag != -1) {
        substitute(A, b, x, size);
    }
    free(s);
    return error_flag;
}

// Generate random vector with length of size
void initialData(double* v, int size) {
    int i;
    double range = 9.0;
    double min = 1.0;
    for (i = 0; i < size; i++) {
        v[i] = min + (rand() / (RAND_MAX / range));
    }
}

// Check if two vectors are close
void checkResult(double* hostRef, double* gpuRef, int size) {
    double epsilon = 1.0E-8;
    int match = 1;
    int i;
    for (i = 0; i < size; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("\thost %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) {
        printf("Arrays match.\n\n");
    }
}

void copyArray(double* a, double* b, int size) {
    for (int i = 0; i < size; i++) {
        b[i] = a[i];
    }
}

// Check the solution of Ax = b, return -1 if not match, return 1 if match
int checkSolution(double** A, double* b, double* x, int size, double tol) {
    double result;
    for (int i = 0; i < size; i++) {
        result = b[i];
        for (int j = 0; j < size; j++) {
            result -= A[i][j] * x[j];
        }
        if (result > tol) {
            return -1;
        }
    }
    return 1;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
