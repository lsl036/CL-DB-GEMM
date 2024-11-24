
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the auto-tuner to tune the xgemm OpenCL kernels.
//
// =================================================================================================

#include "tuning/kernels/myxgemm.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    // case clblast::Precision::kHalf: clblast::Tuner<half>(argc, argv, V, clblast::MyXgemmGetTunerDefaults, clblast::MyXgemmGetTunerSettings<half>, clblast::MyXgemmTestValidArguments<half>, clblast::MyXgemmSetConstraints, clblast::MyXgemmComputeLocalMemSize<half>, clblast::MyXgemmSetArguments<half>); break;
    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::MyXgemmGetTunerDefaults, clblast::MyXgemmGetTunerSettings<float>, clblast::MyXgemmTestValidArguments<float>, clblast::MyXgemmSetConstraints, clblast::MyXgemmComputeLocalMemSize<float>, clblast::MyXgemmSetArguments<float>); break;
    // case clblast::Precision::kDouble: clblast::Tuner<double>(argc, argv, V, clblast::MyXgemmGetTunerDefaults, clblast::MyXgemmGetTunerSettings<double>, clblast::MyXgemmTestValidArguments<double>, clblast::MyXgemmSetConstraints, clblast::MyXgemmComputeLocalMemSize<double>, clblast::MyXgemmSetArguments<double>); break;
    // case clblast::Precision::kComplexSingle: clblast::Tuner<float2>(argc, argv, V, clblast::MyXgemmGetTunerDefaults, clblast::MyXgemmGetTunerSettings<float2>, clblast::MyXgemmTestValidArguments<float2>, clblast::MyXgemmSetConstraints, clblast::MyXgemmComputeLocalMemSize<float2>, clblast::MyXgemmSetArguments<float2>); break;
    // case clblast::Precision::kComplexDouble: clblast::Tuner<double2>(argc, argv, V, clblast::MyXgemmGetTunerDefaults, clblast::MyXgemmGetTunerSettings<double2>, clblast::MyXgemmTestValidArguments<double2>, clblast::MyXgemmSetConstraints, clblast::MyXgemmComputeLocalMemSize<double2>, clblast::MyXgemmSetArguments<double2>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  try {
    // 先看看V = 2的情况
    printf("* (2/4) Tuning main MYGEMM kernel (GEMMK == 0) for random parameters out of larger set\n\n");
    StartVariation<2>(argc, argv);

    printf("* (1/4) Tuning main MYGEMM kernel (GEMMK == 0) for fixed set of parameters\n\n");
    StartVariation<1>(argc, argv);
    
    // printf("* (3/4) Tuning secondary GEMM kernel (GEMMK == 1) for fixed set of parameters\n\n");
    // StartVariation<11>(argc, argv);
    // printf("* (4/4) Tuning secondary GEMM kernel (GEMMK == 1) for random parameters out of larger set\n\n");
    // StartVariation<12>(argc, argv);
    return 0;
  } catch (...) { return static_cast<int>(clblast::DispatchException()); }
}

// =================================================================================================
