
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the SGEMM routine. It is pure C99 and demonstrates the use of
// the C API to the CLBlast library.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include <clblast_c.h>
#include <cblas.h>

// Rowmajor 
#define A(i, j) a[(i)*lda + (j)]
#define B(i, j) b[(i)*ldb + (j)]
#define C(i, j) c[(i)*ldc + (j)]

float compare_matrices(int m, int n, float *a, int lda, float *b, int ldb);
void random_matrix(int m, int n, float *a, int lda);
// =================================================================================================

// Example use of the single-precision routine SGEMM
int main(void) {

  // OpenCL platform/device settings
  const size_t platform_id = 1;
  const size_t device_id = 0;

  // Example SGEMM arguments
  // const size_t m = 128;
  // const size_t n = 64;
  // const size_t k = 512;
  const size_t m = 4096;
  const size_t n = 4096;
  const size_t k = 4096;

  // 检测正确性的settings
  // const int warm_up_ = 0;
  // const int loop_cyc = 1;
  const int warm_up_ = 1;
  const int loop_cyc = 10;

  const float alpha = 0.7f;
  const float beta = 1.0f;
  const size_t a_ld = k;
  const size_t b_ld = n;
  const size_t c_ld = n;

  double timeStart, timeEnd;
  struct timeval tv;
  double timeDuration;

  // Initializes the OpenCL platform
  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
  clGetPlatformIDs(num_platforms, platforms, NULL);
  cl_platform_id platform = platforms[platform_id];

  size_t platform_name_size;
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &platform_name_size);
  char* platform_name = (char*)malloc(platform_name_size);
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);
  printf("Platform Name: %s\n", platform_name);
  free(platform_name);

  // Initializes the OpenCL device
  cl_uint num_devices;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
  cl_device_id device = devices[device_id];

  size_t device_name_size;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_name_size);
  char* device_name = (char*)malloc(device_name_size);
  clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name, NULL);
  printf("Device Name: %s\n", device_name);
  free(device_name);

  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
  cl_event event = NULL;

  // Populate host matrices with some example data
  float* host_a = (float*)malloc(sizeof(float)*m*k);
  float* host_b = (float*)malloc(sizeof(float)*n*k);
  float* host_c = (float*)malloc(sizeof(float)*m*n);
  float* result_c = (float*)malloc(sizeof(float)*m*n);
  for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
  for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
  for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }

  // Copy the matrices to the device
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*sizeof(float), NULL, NULL);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*sizeof(float), NULL, NULL);
  cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, NULL);
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*k*sizeof(float), host_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*k*sizeof(float), host_b, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 0, NULL, NULL);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, host_a, a_ld, host_b, b_ld, beta, host_c, c_ld);

  CLBlastStatusCode status;
  if (warm_up_){
    status = CLBlastSgemm(CLBlastLayoutRowMajor,
                          CLBlastTransposeNo, CLBlastTransposeNo,
                          m, n, k,
                          alpha,
                          device_a, 0, a_ld,
                          device_b, 0, b_ld,
                          beta,
                          device_c, 0, c_ld,
                          &queue, &event);
    if (status == CLBlastSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
  }


  gettimeofday(&tv, NULL);
  timeStart = tv.tv_sec + tv.tv_usec / 1000000.0;
  // Call the SGEMM routine.
  // CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
  //                                         CLBlastTransposeNo, CLBlastTransposeNo,
  //                                         m, n, k,
  //                                         alpha,
  //                                         device_a, 0, a_ld,
  //                                         device_b, 0, b_ld,
  //                                         beta,
  //                                         device_c, 0, c_ld,
  //                                         &queue, &event);
  for (size_t i = 0; i < loop_cyc; i++)
  {
    /* code */
  
  
    status = CLBlastSgemm(CLBlastLayoutRowMajor,
                            CLBlastTransposeNo, CLBlastTransposeNo,
                            m, n, k,
                            alpha,
                            device_a, 0, a_ld,
                            device_b, 0, b_ld,
                            beta,
                            device_c, 0, c_ld,
                            &queue, &event);

    // Wait for completion
    if (status == CLBlastSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }
  }
  gettimeofday(&tv, NULL);
  timeEnd = tv.tv_sec + tv.tv_usec / 1000000.0;
  timeDuration = (timeEnd - timeStart)/loop_cyc;

  clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(float), result_c, 0, NULL, NULL);

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed SGEMM in %.3lf ms with status %d\n", timeDuration*1000.0, status);
  printf("Matrix Sizes: M = %zu, N = %zu, K = %zu\n", m, n, k);
  printf("Sgemm Float-points Performance is: %.3lf GFlops \n", 2*m*n*k/timeDuration/1e9);

  // float diff = 0.0;

  // diff = compare_matrices(m, n, result_c, c_ld, host_c, c_ld);
  // if (diff > 0.5f || diff < -0.5f) {
  //     fprintf(stdout, " diff too big: %le\n", diff);
  //     exit(-1);
  // }
  
  // Clean-up
  free(platforms);
  free(devices);
  free(host_a);
  free(host_b);
  free(host_c);
  clReleaseMemObject(device_a);
  clReleaseMemObject(device_b);
  clReleaseMemObject(device_c);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}

// =================================================================================================
float compare_matrices(int m, int n, float *a, int lda, float *b, int ldb) {
    int i, j;
    float max_diff = 0.0, diff;
    int printed = 0;
  
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        diff = abs(A(i, j) - B(i, j));
        max_diff = (diff > max_diff ? diff : max_diff);
        if (10 > printed)
          if (max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d  j %d diff %f  got %f  expect %f ", i, j, max_diff, A(i, j), B(i, j));
            printed += 1;
          }
      }
    }
  
    return max_diff;
}

void random_matrix(int m, int n, float *a, int lda) {
int i, j;

srand(time(0));
for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
#if 1
        A(i, j) = 1.0 * ( rand() % RAND_MAX ) / RAND_MAX * 5;
#else
        A(i, j) = (j - i) % 3;
#endif
}