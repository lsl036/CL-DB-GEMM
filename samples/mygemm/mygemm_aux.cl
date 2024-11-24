// gemm_auxiliary 
// 用于辅助 GEMM.CL历程进行计算，包含 16个参数的默认定义，相应参数的计算，以及向量寄存器长度的配置，InitAccRegisters 接口实现， GlobalToLocalA， GlobalToLocalB, GlobalToPrivateA, GlobalToPrivateB, GlobalToPrivateA2D, GlobalToPrivateB2D 以及相关的shuffle

R"(
#ifndef GEMMK
  #define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
  #define VWM 1      // Vector width of matrices A and C
#endif
#ifndef VWN
  #define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
  #define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
  #define USE_VECTOR_MAD 1      // Unroll (0) or don't (1) unroll the vector MAD manually 这个参数能影响12Gflops
#endif
#ifndef GLOBAL_MEM_FENCE
  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance 必须为0，如果开启 1 会掉15Gflops
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
  #define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
  #pragma OPENCL EXTENSION cl_intel_subgroups: enable
  #define SUBGROUP_SIZE 8              // Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
  #if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    #define SUBGROUP_SIZE 32            // Assumes subgroup size is always 32 on NVIDIA GPUs
  #endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
  #undef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#elif VWN == 16
    typedef real16 realN;
#endif

// INIT REG
//  =================================================================================================

// Initializes the accumulation registers to zero
INLINE_FUNC realM InitAccRegisters() {
  realM result;
  #if VWM == 1
    SetToZero(result);
  #elif VWM == 2
    SetToZero(result.x);
    SetToZero(result.y);
  #elif VWM == 4
    SetToZero(result.x);
    SetToZero(result.y);
    SetToZero(result.z);
    SetToZero(result.w);
  #elif VWM == 8
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
  #elif VWM == 16
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
    SetToZero(result.s8);
    SetToZero(result.s9);
    SetToZero(result.sA);
    SetToZero(result.sB);
    SetToZero(result.sC);
    SetToZero(result.sD);
    SetToZero(result.sE);
    SetToZero(result.sF);
  #endif
  return result;
}

// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResults(__global realM* cgm, realM c_value, const int _mi, const int _ni,
                              const int kSizeM, const real alpha, const real beta) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  #if STRN == 0
    int ng = _ni + get_local_id(1)*NWI;
  #elif STRN == 1
    int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;
  #endif
  int idm = mg + GetGroupID0() * (MWG/VWM);
  int idn = ng + GetGroupID1() * NWG;
  int index = idn*(kSizeM/VWM) + idm;

  realM result;
  realM xval = c_value;

  // The final multiplication with alpha (in case beta == 0)
  if (IsZero(beta)) {
    #if VWM == 1
      Multiply(result, alpha, xval);
    #elif VWM == 2
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
    #elif VWM == 4
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
      Multiply(result.z, alpha, xval.z);
      Multiply(result.w, alpha, xval.w);
    #elif VWM == 8
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
    #elif VWM == 16
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
      Multiply(result.s8, alpha, xval.s8);
      Multiply(result.s9, alpha, xval.s9);
      Multiply(result.sA, alpha, xval.sA);
      Multiply(result.sB, alpha, xval.sB);
      Multiply(result.sC, alpha, xval.sC);
      Multiply(result.sD, alpha, xval.sD);
      Multiply(result.sE, alpha, xval.sE);
      Multiply(result.sF, alpha, xval.sF);
    #endif
  }

  // The final multiplication with alpha and the addition with beta*C
  else {
    realM yval = cgm[index];
    #if VWM == 1
      AXPBY(result, alpha, xval, beta, yval);
    #elif VWM == 2
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
    #elif VWM == 4
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
      AXPBY(result.z, alpha, xval.z, beta, yval.z);
      AXPBY(result.w, alpha, xval.w, beta, yval.w);
    #elif VWM == 8
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
    #elif VWM == 16
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
      AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
      AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
      AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
      AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
      AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
      AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
      AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
      AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
    #endif
  }
  cgm[index] = result;
}

// GLOBAL TO LOCAL
//=======================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm, const int kSizeM, const int tid, const int kwg) 
{
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = _mia + la0*(MWA/VWM);
      #elif STRM == 1 // strided 访存
        int mg = la0 + _mia*MDIMA;  // M方向，每个线程跨越MDIMA个取 m-group
      #endif

      // Computes the indices for the global memory
      int kg = _kia + la1*KWA;      // K方向是每个线程连续KWA个取 k-group
      int idm = mg + GetGroupID0() * (MWG/VWM); // global里的元素对应的M维度序号(行号) ID-m
      int idk = kg + kwg;   // global里元素对应的K维度序号(列号)  ID-k

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm, const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = _nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + _nib*NDIMB;  // group内的n维度
      #endif

      // Computes the indices for the global memory
      int kg = _kib + lb1*KWB;    // group 内的k维度
      int idn = ng + GetGroupID1() * (NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];   
    }
  }
}
#endif

// LOCAL TO PRIVATE
// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
INLINE_FUNC realM LocalToPrivateA(LOCAL_PTR realM* alm, const int _mi, const int kg) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1 // 跨DIMC访问以减少bank冲突 
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  return alm[kg*(MWG/VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC realN LocalToPrivateB(LOCAL_PTR realN* blm, const int _ni, const int kg) {
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif
  return blm[kg*(NWG/VWN) + ng];
}
#endif

// GLOBAL TO PRIVATE
//=================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,
                                   const int kSizeM, const int idk, const int kwg) {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM); //竖着连续取Mi
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;     // 横着 跳跃取Mi
  #endif

  // Computes the indices for the global memory
  int idm = mg + GetGroupID0() * (MWG/VWM);

  // Loads the data from global memory (not transposed) and stores into registers
  return agm[idk*(kSizeM/VWM) + idm];   // ksizeM 是 M 维度
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
INLINE_FUNC realN GlobalToPrivateB(const __global realN* restrict bgm, const int _ni,
                                   const int kSizeN, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif

  // Computes the indices for the global memory
  int idn = ng + GetGroupID1() * (NWG/VWN);

  // Loads the data from global memory (transposed) and stores into registers
  return bgm[idk*(kSizeN/VWN) + idn];
}
#endif

// Global To Private2D
// =================================================================================================
#if GEMMK == 1

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix for kernel 1.
INLINE_FUNC realN GlobalToPrivateA2D(const __global real* restrict a_ptr, const int tid_y, const int _ni,
                                     const int kSizeK, const int idk, const int _ki) {
  #if PRECISION == 3232 || PRECISION == 6464
    const int a_index = (tid_y * NWI + _ni) * (kSizeK / VWN) + idk / VWN + _ki;
    const __global realN* restrict agm = (const __global realN* restrict) a_ptr;
    return agm[a_index];
  #else
    const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * VWN;
    #if VWN == 1
      return a_ptr[a_index];
    #elif VWN == 2
      return vload2(0, a_ptr + a_index);
    #elif VWN == 4
      return vload4(0, a_ptr + a_index);
    #elif VWN == 8
      return vload8(0, a_ptr + a_index);
    #elif VWN == 16
      return vload16(0, a_ptr + a_index);
    #endif
  #endif
}

// Same as above, but now for the B input matrix
INLINE_FUNC realM GlobalToPrivateB2D(const __global real* restrict b_ptr, const int tid_x, const int _mi,
                                     const int kSizeN, const int idk, const int _ki) {
  #if PRECISION == 3232 || PRECISION == 6464
    const int b_index = (idk + _ki) * (kSizeN / VWM) + tid_x * (MWI / VWM) + _mi;
    const __global realM* restrict bgm = (const __global realM* restrict) b_ptr;
    return bgm[b_index];
  #else
    const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * VWM;
    #if VWM == 1
      return b_ptr[b_index];
    #elif VWM == 2
      return vload2(0, b_ptr + b_index);
    #elif VWM == 4
      return vload4(0, b_ptr + b_index);
    #elif VWM == 8
      return vload8(0, b_ptr + b_index);
    #elif VWM == 16
      return vload16(0, b_ptr + b_index);
    #endif
  #endif
}

#endif

// A common interface for subgroup functions

#if USE_SUBGROUP_SHUFFLING == 1

    INLINE_FUNC int clblast_get_sub_group_local_id() {

    // Intel extension 
#if SUBGROUP_SHUFFLING_INTEL == 1
    return get_sub_group_local_id();
    
    // Nvidia inline PTX
#elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    int ret;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret) );
    return ret;
    #endif 
    }

    INLINE_FUNC realN clblast_sub_group_shuffle(realN reg, int src) {

    // Intel extension 
    #if SUBGROUP_SHUFFLING_INTEL == 1
    return intel_sub_group_shuffle(reg, src);
    
    // Nvidia inline PTX
    // Volta and later requires .sync shuffle instructions with an extra mask arg
    #elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    realN ret;
        #if SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret): "f"(reg), "r"(src));
        #else
        asm volatile("shfl.idx.b32 %0, %1, %2, 0x1f;" : "=f"(ret): "f"(reg), "r"(src));
        #endif
    return ret;
#endif
    }
#endif

);