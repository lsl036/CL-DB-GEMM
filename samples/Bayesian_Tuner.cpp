#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>

// Includes the OpenCL tuner library
#include "clblast.h"
#include "../src/utilities/utilities.hpp"
#include "../src/utilities/compile.hpp"
#include "../src/utilities/timing.hpp"
#include "../src/tuning/configurations.hpp"
#include "../src/tuning/kernels/xgemm.hpp"

// #define DEBUG

// This sample is used for Runkernel.py to iteratively running GEMM under different parameters settings.
// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) {
  return ((a/b)*b == a) ? true : false;
};

#if defined(_WIN32)
    const std::string kPrintError = "";
    const std::string kPrintSuccess = "";
    const std::string kPrintMessage = "";
    const std::string kPrintEnd = "";
#else
    const std::string kPrintError = "\x1b[31m";
    const std::string kPrintSuccess = "\x1b[32m";
    const std::string kPrintMessage = "\x1b[1m";
    const std::string kPrintEnd = "\x1b[0m";
#endif

const std::vector<std::string> options = {"GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"};

const int V = 2;

struct Hypers_GEMM {
  size_t GEMMK = 0;
  size_t KREG = 2;
  size_t KWG = 32;
  size_t KWI = 4;
  size_t MDIMA = 16;
  size_t MDIMC = 32;
  size_t MWG = 32;
  size_t NDIMB = 64;
  size_t NDIMC = 16;
  size_t NWG = 128;
  size_t SA = 1;
  size_t SB = 1;
  size_t STRM = 0;
  size_t STRN = 0;
  size_t VWM = 1;
  size_t VWN = 2;
};
void print_separator(const size_t parameters_size) {
  printf("x------x-------x");
  for (auto i = size_t{0}; i < parameters_size; ++i) { printf("-----"); }
  printf("-x-----------------x-----------------x----------------x--------------x--------x-------------------x\n");
}

clblast::TunerSettings BayesianSettings(const int V, const clblast::Arguments<float> &args, Hypers_GEMM hyper_para){
  auto settings = clblast::TunerSettings();

  // Identification of the kernel
  settings.kernel_family = "xgemm_" + clblast::ToString(V);
  settings.kernel_name = "Xgemm";
  settings.sources = (V == 11 || V == 12) ? "#define GEMMK 1" : "#define GEMMK 0";
  settings.sources +=
#include "../src/kernels/level3/xgemm_part1.opencl"
#include "../src/kernels/level3/xgemm_part2.opencl"
  ;
  settings.sources +=
#include "../src/kernels/level3/xgemm_part3.opencl"
#include "../src/kernels/level3/xgemm_part4.opencl"
  ;
  // Buffer sizes
  settings.size_a = args.m * args.k;
  settings.size_b = args.n * args.k;
  settings.size_c = args.m * args.n;

  // Inputs and outputs IDs (X:0, Y:1, A:2, B:3, C:4, temp:5)
  settings.inputs = {2, 3, 4};
  settings.outputs = {4};

  // Sets the base thread configuration
  settings.global_size = {args.m, args.n};
  settings.global_size_ref = settings.global_size;
  settings.local_size = {1, 1};
  settings.local_size_ref = {8, 8};

  // Transforms the thread configuration based on the parameters
  settings.mul_local = {{"MDIMC", "NDIMC"}};
  settings.mul_global = {{"MDIMC", "NDIMC"}};
  settings.div_global = {{"MWG", "NWG"}};

  settings.parameters={
    {"GEMMK",{hyper_para.GEMMK}},
    {"KREG",{hyper_para.KREG}},
    {"KWG",{hyper_para.KWG}},
    {"KWI",{hyper_para.KWI}},
    {"MDIMA",{hyper_para.MDIMA}},
    {"MDIMC",{hyper_para.MDIMC}},
    {"MWG",{hyper_para.MWG}},
    {"NDIMB",{hyper_para.NDIMB}},
    {"NDIMC",{hyper_para.NDIMC}},
    {"NWG",{hyper_para.NWG}},
    {"SA",{hyper_para.SA}},
    {"SB",{hyper_para.SB}},
    {"STRM",{hyper_para.STRM}},
    {"STRN",{hyper_para.STRN}},
    {"VWM",{hyper_para.VWM}},
    {"VWN",{hyper_para.VWN}}
  };

  // Describes how to compute the performance metrics
  if((args.precision == clblast::Precision::kComplexSingle) || (args.precision == clblast::Precision::kComplexDouble)) {
    // complex flops
    settings.metric_amount = args.m * args.n * (8 * args.k - 2);
  } else {
    // scalar flops
    settings.metric_amount = args.m * args.n * (2 * args.k - 1);
  }
  settings.performance_unit = "GFLOPS";

  return settings;
}

double Res_Performance = 0.0;

int main(int argc, char** argv) {
  constexpr auto kSeed = 42; // fixed seed for reproducibility

  const clblast::TunerDefaults defaults = clblast::XgemmGetTunerDefaults(V);
  
  // 获取 输入的参数
  auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);

  auto help = std::string{"* Options given/available:\n"};

  auto hyper_para = Hypers_GEMM{};
  hyper_para.GEMMK = clblast::GetArgument(command_line_args, help, "GEMMK", size_t{0});
  hyper_para.KREG = clblast::GetArgument(command_line_args, help, "KREG", size_t{1});
  hyper_para.KWG = clblast::GetArgument(command_line_args, help, "KWG", size_t{32});
  hyper_para.KWI = clblast::GetArgument(command_line_args, help, "KWI", size_t{2});
  hyper_para.MDIMA = clblast::GetArgument(command_line_args, help, "MDIMA", size_t{8});
  hyper_para.MDIMC = clblast::GetArgument(command_line_args, help, "MDIMC", size_t{8});
  hyper_para.MWG = clblast::GetArgument(command_line_args, help, "MWG", size_t{32});
  hyper_para.NDIMB = clblast::GetArgument(command_line_args, help, "NDIMB", size_t{16});
  hyper_para.NDIMC = clblast::GetArgument(command_line_args, help, "NDIMC", size_t{16});
  hyper_para.NWG = clblast::GetArgument(command_line_args, help, "NWG", size_t{64});
  hyper_para.SA = clblast::GetArgument(command_line_args, help, "SA", size_t{1});
  hyper_para.SB = clblast::GetArgument(command_line_args, help, "SB", size_t{1});
  hyper_para.STRM = clblast::GetArgument(command_line_args, help, "STRM", size_t{0});
  hyper_para.STRN = clblast::GetArgument(command_line_args, help, "STRN", size_t{0});
  hyper_para.VWM = clblast::GetArgument(command_line_args, help, "VWM", size_t{4});
  hyper_para.VWN = clblast::GetArgument(command_line_args, help, "VWN", size_t{4});

#ifdef DEBUG
  std::cout << "========== Args infos ==========" << std::endl;
  std::cout << options[0] << " = " << hyper_para.GEMMK << std::endl;
  std::cout << options[1] << " = " << hyper_para.KREG << std::endl;
  std::cout << options[2] << " = " << hyper_para.KWG << std::endl;
  std::cout << options[3] << " = " << hyper_para.KWI << std::endl;
  std::cout << options[4] << " = " << hyper_para.MDIMA << std::endl;
  std::cout << options[5] << " = " << hyper_para.MDIMC << std::endl;
  std::cout << options[6] << " = " << hyper_para.MWG << std::endl;
  std::cout << options[7] << " = " << hyper_para.NDIMB << std::endl;
  std::cout << options[8] << " = " << hyper_para.NDIMC << std::endl;
  std::cout << options[9] << " = " << hyper_para.NWG << std::endl;
  std::cout << options[10] << " = " << hyper_para.SA << std::endl;
  std::cout << options[11] << " = " << hyper_para.SB << std::endl;
  std::cout << options[12] << " = " << hyper_para.STRM << std::endl;
  std::cout << options[13] << " = " << hyper_para.STRN << std::endl;
  std::cout << options[14] << " = " << hyper_para.VWM << std::endl;
  std::cout << options[15] << " = " << hyper_para.VWN << std::endl;
#endif
  
  auto args = clblast::Arguments<float>{};
  args.platform_id = clblast::GetArgument(command_line_args, help, clblast::kArgPlatform, clblast::ConvertArgument(std::getenv("CLBLAST_PLATFORM"), size_t{0}));
  args.device_id   = clblast::GetArgument(command_line_args, help, clblast::kArgDevice, clblast::ConvertArgument(std::getenv("CLBLAST_DEVICE"), size_t{0}));
  args.precision   = clblast::GetArgument(command_line_args, help, clblast::kArgPrecision, clblast::Precision::kSingle);
  for (auto &o: defaults.options) {
    if (o == clblast::kArgM)        { args.m        = clblast::GetArgument(command_line_args, help, clblast::kArgM, defaults.default_m); }
    if (o == clblast::kArgN)        { args.n        = clblast::GetArgument(command_line_args, help, clblast::kArgN, defaults.default_n); }
    if (o == clblast::kArgK)        { args.k        = clblast::GetArgument(command_line_args, help, clblast::kArgK, defaults.default_k); }
    if (o == clblast::kArgChannels)   { args.channels    = clblast::GetArgument(command_line_args, help, clblast::kArgChannels, defaults.channels); }
    if (o == clblast::kArgHeight)     { args.height      = clblast::GetArgument(command_line_args, help, clblast::kArgHeight, defaults.height); }
    if (o == clblast::kArgWidth)      { args.width       = clblast::GetArgument(command_line_args, help, clblast::kArgWidth, defaults.width); }
    if (o == clblast::kArgKernelH)    { args.kernel_h    = clblast::GetArgument(command_line_args, help, clblast::kArgKernelH, defaults.kernel_h); }
    if (o == clblast::kArgKernelW)    { args.kernel_w    = clblast::GetArgument(command_line_args, help, clblast::kArgKernelW, defaults.kernel_w); }
    if (o == clblast::kArgNumKernels) { args.num_kernels = clblast::GetArgument(command_line_args, help, clblast::kArgNumKernels, defaults.num_kernels); }
    if (o == clblast::kArgAlpha)      { args.alpha       = clblast::GetArgument(command_line_args, help, clblast::kArgAlpha, clblast::GetScalar<float>()); }
    if (o == clblast::kArgBeta)       { args.beta        = clblast::GetArgument(command_line_args, help, clblast::kArgBeta, clblast::GetScalar<float>()); }
    if (o == clblast::kArgBatchCount) { args.batch_count = clblast::GetArgument(command_line_args, help, clblast::kArgBatchCount, defaults.default_batch_count); }
  }
  args.fraction = clblast::GetArgument(command_line_args, help, clblast::kArgFraction, defaults.default_fraction);
  args.num_runs = clblast::GetArgument(command_line_args, help, clblast::kArgNumRuns, defaults.default_num_runs);
  const auto max_l2_norm = clblast::GetArgument(command_line_args, help, clblast::kArgMaxL2Norm, 1.0e-4);
  
#ifdef DEBUG
  printf("%s\n", help.c_str());
  std::cout << "args.m = " << args.m << std::endl;
  std::cout << "args.n = " << args.n << std::endl;
  std::cout << "args.k = " << args.k << std::endl;
  std::cout << "max_l2_norm = " << max_l2_norm << std::endl;
#endif
  // 初始化kernel的设定,包含kernel名称,kernel的cl文件,buffer size, 相应的参数组合
  const clblast::TunerSettings settings = BayesianSettings(V, args, hyper_para);

  clblast::XgemmTestValidArguments(V, args);

  // OpenCL 设备配置
  // CLBLASt - settings
  const auto platform = clblast::Platform(args.platform_id);
  const auto device = clblast::Device(platform, args.device_id);
  const auto context = clblast::Context(device);
  auto queue = clblast::Queue(context, device);

  // Tests for validity of the precision and retrieves properties
  if (!clblast::PrecisionSupported<float>(device)) {
    printf("* Unsupported precision, skipping this tuning run\n\n");
    return -1;
  }
  const auto device_type = GetDeviceType(device);
  const auto device_vendor = GetDeviceVendor(device);
  const auto device_architecture = GetDeviceArchitecture(device);
  const auto device_name = GetDeviceName(device);
#ifdef DEBUG
  std::cout << "device_type = " << device_type << std::endl;
  std::cout << "device_vendor = " << device_vendor << std::endl;
  std::cout << "device_architecture = " << device_architecture << std::endl;
  std::cout << "device_name = " << device_name << std::endl;
#endif

  // x, y, A, B, C, temp 
  const auto buffer_sizes = std::vector<size_t>{
    settings.size_x + clblast::kCanarySize, settings.size_y + clblast::kCanarySize,
    settings.size_a + clblast::kCanarySize, settings.size_b + clblast::kCanarySize, settings.size_c + clblast::kCanarySize,
    settings.size_temp + clblast::kCanarySize
  };
    // 创建一个随机数生成器
  std::mt19937 mt(kSeed); // std::mt19937 是伪随机数产生器，用于产生高性能的随机数。
  std::uniform_real_distribution<double> dist(-2.0, 2.0); // [-2,2) 均匀分布中生成随机浮点数
  // 创建输入矩阵
  auto source_buffers = std::vector<std::vector<float>>();
  auto reference_buffers = std::vector<std::vector<float>>();
  auto result_buffers = std::vector<std::vector<float>>();
  auto device_buffers = std::vector<clblast::Buffer<float>>();

  // 随机数赋值
  for (const auto size : buffer_sizes) {
    auto host_buffer = std::vector<float>(size);
    clblast::PopulateVector(host_buffer, mt, dist);    //用[-2,2)的随机数据填充向量
    source_buffers.push_back(host_buffer);
    reference_buffers.push_back(std::vector<float>(size));
    result_buffers.push_back(std::vector<float>(size));
    device_buffers.push_back(clblast::Buffer<float>(context, size));
  }
#ifdef DEBUG
    for (const auto id : settings.outputs) {
      std::cout << "\nC[0] = " << reference_buffers[id][0] << std::endl;
      std::cout << "C[1] = " << reference_buffers[id][1] << std::endl;
      std::cout << "C[2] = " << reference_buffers[id][2] << std::endl;
      std::cout << "C[3] = " << reference_buffers[id][3] << std::endl;
      std::cout << "C[4] = " << reference_buffers[id][4] << std::endl;
    }
#endif
  
  // Sets the tunable parameters and their possible values V==2 的时候 GEMMK == 0
  auto configurations = SetConfigurations(device, settings.parameters, settings.local_size,
                                          settings.mul_local, settings.div_local,
                                          clblast::XgemmSetConstraints(V), clblast::XgemmComputeLocalMemSize<float>(V));
#ifdef DEBUG
  printf("* Found %s%zu configuration(s)%s\n",
         kPrintMessage.c_str(), configurations.size(), kPrintEnd.c_str());

  // Select the search method (full search or a random fraction)
  // if (args.fraction != 0.0 && args.fraction != 1.0) {
  //   const auto new_size = static_cast<size_t>(configurations.size() / args.fraction);
  //   auto rng = std::default_random_engine{};
  //   // 最简单的思路，shuffle然后选前面 0~ fraction 的部分
  //   std::shuffle(std::begin(configurations), std::end(configurations), rng);
  //   configurations.resize(new_size);
  //   printf("* Exploring a random subset of %s%zu configuration(s)%s\n",
  //          kPrintMessage.c_str(), configurations.size(), kPrintEnd.c_str());
  // }

  // Prints information about the parameters
  printf("* Parameters explored: ");
  for (const auto& parameter : settings.parameters) { printf("%s ", parameter.first.c_str()); }
  printf("\n");
#endif

  // Prints the header of the table
  printf("\n");
  printf("|   ID | total |");
  for (auto i = size_t{0}; i < settings.parameters.size() - 1; ++i) { printf("     "); }
  printf("param |      local      |      global     |       compiles |         time | %6s |            status |\n", settings.performance_unit.c_str());
  print_separator(settings.parameters.size());

  // 首先运行一个参考示例来比较 Reference
  try {
    printf("|  ref |     - |");
    for (auto i = size_t{0}; i < settings.parameters.size() - 1; ++i) { printf("     "); }
    printf("    - |");


    // Sets the input
    for (const auto id : settings.inputs) {
      device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
    }

    // Sets the thread configuration
    auto global = settings.global_size_ref;
    auto local = settings.local_size_ref;

    // Make sure that the global worksize is a multiple of the local
    for (auto i=size_t{0}; i<global.size(); ++i) {
      while ((global[i] / local[i]) * local[i] != global[i]) { global[i]++; }
    }
    if (local.size() > 1 && global.size() > 1) {
      printf("%8zu%8zu |%8zu%8zu |", local[0], local[1], global[0], global[1]);
    }
    else {
      printf("%8zu%8d |%8zu%8d |", local[0], 1, global[0], 1);
    }

    // Compiles the kernel
    auto compiler_options = std::vector<std::string>();
    const auto program = clblast::CompileFromSource(settings.sources, args.precision, settings.kernel_name,
                                           device, context, compiler_options, 0);
    auto kernel = clblast::Kernel(program, settings.kernel_name);
    clblast::XgemmSetArguments<float>(V, kernel, args, device_buffers);
    printf("             %sOK%s |", kPrintSuccess.c_str(), kPrintEnd.c_str());

    // Runs the kernel
    const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device,
                                    global, local);
    printf("      - |");
    if (time_ms == -1.0) { throw std::runtime_error("Error in reference implementation"); }

    // Saves the result
    for (const auto id : settings.outputs) {
      device_buffers[id].Read(queue, buffer_sizes[id], reference_buffers[id]);
    }
    printf("      %sreference OK%s |\n", kPrintSuccess.c_str(), kPrintEnd.c_str());
  #ifdef DEBUG
  // 最新测试说明，这个reference算的是错误的, 我们的结果跟Tuner那边是一样的
    for (const auto id : settings.outputs) {
      std::cout << "\nC[0] = " << reference_buffers[id][0] << std::endl;
      std::cout << "C[1] = " << reference_buffers[id][1] << std::endl;
      std::cout << "C[2] = " << reference_buffers[id][2] << std::endl;
      std::cout << "C[3] = " << reference_buffers[id][3] << std::endl;
      std::cout << "C[4] = " << reference_buffers[id][4] << std::endl;
    }
  #endif
  }
  catch (...) {
    const auto status_code = clblast::DispatchExceptionCatchAll(true);
    printf("* Exception caught with status %d while running the reference, aborting\n",
           static_cast<int>(status_code));
    return -1;
  }
  print_separator(settings.parameters.size());

  // try only 1 configuration
  for (auto config_id = size_t{0}; config_id < configurations.size(); ++config_id) {
    try{
      auto configuration = configurations[config_id];
      printf("| %4zu | %5zu |", config_id + 1, configurations.size());
      for (const auto& parameter : settings.parameters)
      {
        // 使用parameter的 string 作为key来输出可行 configuration中的值
        printf("%5zu", configuration.at(parameter.first));
      }
      printf(" |");

      // Sets the input
      for (const auto id : settings.inputs) {  // settings.inputs = {2, 3, 4};
      // 封装的 clEnqueueWriteBuffer 方法
        device_buffers[id].Write(queue, buffer_sizes[id], source_buffers[id]);
      }

      auto global = clblast::SetThreadConfiguration(configuration,settings.global_size, settings.mul_global, settings.div_global);
      auto local = clblast::SetThreadConfiguration(configuration, settings.local_size, settings.mul_local, settings.div_local);

      // Make sure that the global worksize is a multiple of the local
      for (auto i=size_t{0}; i<global.size(); ++i) {
        while ((global[i] / local[i]) * local[i] != global[i]) { global[i]++; }
      }
      if (local.size() > 1 && global.size() > 1) {
        printf("%8zu%8zu |%8zu%8zu |", local[0], local[1], global[0], global[1]);
      }
      else{
        printf("%8zu%8d |%8zu%8d |", local[0], 1, global[0], 1);
      }

      // set the hyper parameters for configuration
      auto kernel_source = std::string{""};
      for (const auto &parameter : configuration){
        kernel_source += "#define " + parameter.first + " " + clblast::ToString(parameter.second) + "\n";
      }
#ifdef DEBUG
      std::cout << kernel_source << std::endl;
#endif
      kernel_source += settings.sources;
#ifdef DEBUG
      // std::cout << kernel_source << std::endl;
#endif

      // Compiles the kernel
      const auto start_time = std::chrono::steady_clock::now();
      auto compiler_options = std::vector<std::string>();
      const auto program = CompileFromSource(kernel_source, args.precision, settings.kernel_name, device, context, compiler_options, 0, true);
      auto kernel = clblast::Kernel(program, settings.kernel_name);
      const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
      const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
      printf("   %sOK%s  %5.0lf ms |", kPrintSuccess.c_str(), kPrintEnd.c_str(), timing);

      // Runs the kernel
      clblast::XgemmSetArguments<float>(V, kernel, args, device_buffers);
      const auto time_ms = TimeKernel(args.num_runs, kernel, queue, device, global, local);

      // Kernel run was not successful
      if (time_ms == -1.0) {
        printf("      - |");
        printf("   %sinvalid config.%s |", kPrintError.c_str(), kPrintEnd.c_str());
        printf(" <-- skipping\n");
        continue;
      }

      // Compares the results
      auto l2_error = 0.0;
      for (const auto id : settings.outputs) {
        device_buffers[id].Read(queue, buffer_sizes[id], result_buffers[id]);
#ifdef DEBUG
        for (const auto id : settings.outputs) {
          std::cout << "\nC[0] = " << result_buffers[id][0] << std::endl;
          std::cout << "C[1] = " << result_buffers[id][1] << std::endl;
          std::cout << "C[2] = " << result_buffers[id][2] << std::endl;
          std::cout << "C[3] = " << result_buffers[id][3] << std::endl;
          std::cout << "C[4] = " << result_buffers[id][4] << std::endl;
        }
#endif
        for (auto index = size_t{0}; index<buffer_sizes[id]; ++index) {
          const auto diff = clblast::SquaredDifference(result_buffers[id][index], reference_buffers[id][index]);
          l2_error += diff;
        }
        l2_error /= static_cast<double>(buffer_sizes[id]);  // L2 为均方误差
        if (std::isnan(l2_error) || l2_error > max_l2_norm) {
          printf("      - |");
          printf(" %sL2 error %8.2e%s |", kPrintError.c_str(), l2_error, kPrintEnd.c_str());
          throw std::runtime_error("L2 error too large");
        }
      }

      // All was OK
      configuration["PRECISION"] = static_cast<size_t>(args.precision);
      // results.push_back(TuningResult{settings.kernel_name, time_ms, configuration});
      Res_Performance = settings.metric_amount / (time_ms * 1.0e6);
      printf(" %6.4lf |", settings.metric_amount / (time_ms * 1.0e6));  // Gflops
      printf("     %sresults match%s |\n", kPrintSuccess.c_str(), kPrintEnd.c_str());
    }
    catch(clblast::CLCudaAPIBuildError&){
      const auto status_code = clblast::DispatchExceptionCatchAll(true);
      printf("  %scompilation error: %5d%s     |",
             kPrintError.c_str(), static_cast<int>(status_code), kPrintEnd.c_str());
      printf("      - |                 - | <-- skipping\n");
    }
    catch (...) {
      const auto status_code = clblast::DispatchExceptionCatchAll(true);
      if (status_code != clblast::StatusCode::kUnknownError) {
        printf("   %serror code %d%s |",
               kPrintError.c_str(), static_cast<int>(status_code), kPrintEnd.c_str());
      }
      printf(" <-- skipping\n");
    }
  }
#ifdef DEBUG
  printf("%lf\n",Res_Performance);
#endif
  return 0;
}