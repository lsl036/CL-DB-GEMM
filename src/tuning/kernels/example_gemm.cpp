
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

#include "tuning/kernels/xgemm.hpp"

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {

    case clblast::Precision::kSingle: clblast::Tuner<float>(argc, argv, V, clblast::XgemmGetTunerDefaults, clblast::XgemmGetTunerSettings<float>, clblast::XgemmTestValidArguments<float>, clblast::XgemmSetConstraints, clblast::XgemmComputeLocalMemSize<float>, clblast::XgemmSetArguments<float>); break;
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {
  try {
    printf("* (1/1) Tuning main GEMM kernel (GEMMK == 0) by Bayesian Optimization\n\n");
    StartVariation<3>(argc, argv);
    
    return 0;
  } catch (...) { return static_cast<int>(clblast::DispatchException()); }
}

// =================================================================================================
