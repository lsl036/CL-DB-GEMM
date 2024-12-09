
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

// 0522 已经实现了完全的ping-pong, 且结果正确. 需要进行调参尝试
INLINE_FUNC void MyXgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                           const __global realM* restrict agm, const __global realN* restrict bgm,
                           __global realM* cgm, const real alpha, const real beta
                          //  #if SA == 1 && SB == 1
                             , LOCAL_PTR realM* alm
                             , LOCAL_PTR realN* blm
                          //  #elif SA == 1
                          //    , LOCAL_PTR realM* alm
                          //  #elif SB == 1
                          //    , LOCAL_PTR realN* blm
                          //  #endif
                           ) 
{
  // Allocates workitem-private memory (registers)
  #if GEMMK == 0
    #pragma promote_to_registers
    realM apm[2][MWI/VWM]; //  MWI=2 VWM=1   apm[2][2]
    #pragma promote_to_registers
    realN bpm[2][NWI/VWN]; //  NWI=4 VWN=2   bpm[2][2]
  #endif

  #pragma promote_to_registers
  realM cpm[NWI*(MWI/VWM)]; // NWI * MWI (4 * 2)

  // Combined thread identifier (volatile to disable caching)
  volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  
  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters(); // col major
    }
  }

  // 加载第一批数据到 local memory, 再load 到寄存器
  
  // 中间寄存器
  #pragma promote_to_registers
  realM a_ldg_reg[MWA/VWM * KWA];  // RX550 上 2/1 * 2
  realN b_ldg_reg[NWB/VWN * KWB];  // RX550 上 4/2 * 4

  int mg, kg_a, kg_b, ng, idm, idk_a, idk_b, idn;
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;

  #if STRM == 1
  const int agm_offset = la1 * KWA * (kSizeM/VWM) + GetGroupID0() * (MWG/VWM) + la0;
  const int alm_offset = la1*KWA*(MWG/VWM) + la0;
  #elif STRM == 0
  const int agm_offset = la1 * KWA * (kSizeM/VWM) + GetGroupID0() * (MWG/VWM) + la0*(MWA/VWM);
  const int alm_offset = la1*KWA*(MWG/VWM) + la0*(MWA/VWM);
  #endif

  #if STRN == 1
  const int bgm_offset = lb1 * KWB * (kSizeN/VWN) + GetGroupID1() * (NWG/VWN) + lb0;
  const int blm_offset = lb1*KWB*(NWG/VWN) + lb0;
  #elif STRN == 0
  const int bgm_offset = lb1 * KWB * (kSizeN/VWN) + GetGroupID1() * (NWG/VWN) + lb0*(NWB/VWN);
  const int blm_offset = lb1*KWB*(NWG/VWN) + lb0*(NWB/VWN);
  #endif

  // global --> reg
  // from Global A to a_ldg_reg
  int kwg = 0;

  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) { //  0 ~ 2 
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) { //  0 ~ 2
    #if STRM == 1
      a_ldg_reg[_kia * MWA/VWM + _mia] = agm[agm_offset + (_kia)*(kSizeM/VWM) + _mia * MDIMA];
    #elif STRM == 0
      a_ldg_reg[_kia * MWA/VWM + _mia] = agm[agm_offset + (_kia)*(kSizeM/VWM) + _mia];
    #endif
    }
  }
  // from Global B to b_ldg_reg
  
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {
    #if STRN == 1
      b_ldg_reg[_kib * NWB/VWN + _nib] = bgm[bgm_offset + (_kib)*(kSizeN/VWN) + _nib * NDIMB];
    #elif STRN == 0
      b_ldg_reg[_kib * NWB/VWN + _nib] = bgm[bgm_offset + (_kib)*(kSizeN/VWN) + _nib];
    #endif
    }
  }

  // reg --> Local
  // from a_ldg_reg to alm
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {
    #if STRM == 1
      alm[alm_offset + (_kia)*(MWG/VWM) + _mia * MDIMA] = a_ldg_reg[_kia * MWA/VWM  + _mia];
    #elif STRM == 0
      alm[alm_offset + (_kia)*(MWG/VWM) + _mia] = a_ldg_reg[_kia * MWA/VWM  + _mia];
    #endif
    }
  }

  // from b_ldg_reg to blm
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {
    #if STRN == 1
      blm[blm_offset + _kib*(NWG/VWN) + _nib*NDIMB] = b_ldg_reg[_kib * NWB/VWN + _nib];
    #elif STRN == 0
      blm[blm_offset + _kib*(NWG/VWN) + _nib] = b_ldg_reg[_kib * NWB/VWN + _nib];
    #endif
    }
  }

  // Global --> Local
  // GlobalToLocalA(agm, alm, kSizeM, tid, 0);
  // GlobalToLocalB(bgm, blm, kSizeN, tid, 0);
  barrier(CLK_LOCAL_MEM_FENCE);

  // 写进 shared mem中的 idx
  int write_stage_idx = 1;

  #if STRM == 1
    const int aLtoP_offset = get_local_id(0);
  #elif STRM == 0
    const int aLtoP_offset = get_local_id(0)*(MWI/VWM);
  #endif

  // Local --> Private memory 加载第一次计算的apm, bpm
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
  #if STRM == 1
    apm[0][_mi] = alm[aLtoP_offset + 0*(MWG/VWM) + _mi*MDIMC];
  #elif STRM == 0
    apm[0][_mi] = alm[aLtoP_offset + 0*(MWG/VWM) + _mi];
  #endif
  }

  #if STRN == 1
    const int bLtoP_offset = get_local_id(1);
  #elif STRN == 0
    const int bLtoP_offset = get_local_id(1)*(NWI/VWN);
  #endif

  #pragma unroll
  for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
  #if STRN == 1
    bpm[0][_ni] = blm[bLtoP_offset + 0*(NWG/VWN) + _ni*NDIMC];
  #elif STRN == 0
    bpm[0][_ni] = blm[bLtoP_offset + 0*(NWG/VWN) + _ni];
  #endif
  }

  // 执行 ping-pong 运算
  do
  {
    kwg += KWG * KREG;

    // 预取下一次的 Global数据到 中间寄存器 global --> reg

    // from Global A to a_ldg_reg
    #pragma unroll
    for (int _mia = 0; _mia < MWA/VWM; _mia += 1) { //  0 ~ 2 
      #pragma unroll
      for (int _kia = 0; _kia < KWA; _kia += 1) { //  0 ~ 2
      #if STRM == 1
        a_ldg_reg[_kia * MWA/VWM + _mia] = agm[agm_offset + (_kia + kwg)*(kSizeM/VWM) + _mia * MDIMA];
      #elif STRM == 0
        a_ldg_reg[_kia * MWA/VWM + _mia] = agm[agm_offset + (_kia + kwg)*(kSizeM/VWM) + _mia];
      #endif
      }
    }

    // from Global B to b_ldg_reg
    #pragma unroll
    for (int _kib = 0; _kib < KWB; _kib += 1) {
      #pragma unroll
      for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {
      #if STRN == 1
        b_ldg_reg[_kib * NWB/VWN + _nib] = bgm[bgm_offset + (_kib + kwg)*(kSizeN/VWN) + _nib * NDIMB];
      #elif STRN == 0
        b_ldg_reg[_kib * NWB/VWN + _nib] = bgm[bgm_offset + (_kib + kwg)*(kSizeN/VWN) + _nib];
      #endif
      }
    }
    
    int load_stage_idx = write_stage_idx ^ 1;
    
    // 加载 j = 0 ~ 14
    #pragma unroll
    for (int j = 0; j < (KWG - 1)* KREG; j += KREG)
    {
    
      #pragma unroll
      for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
        const realM aval = apm[j % 2][_mi];
        #pragma unroll
        for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
          #if  VWN == 1
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni]);
          #elif VWN == 2
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].x);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].y);
          #elif VWN == 4
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].x);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].y);
            cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].z);
            cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].w);
          #elif VWN == 8
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s0);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s1);
            cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s2);
            cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s3);
            cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s4);
            cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s5);
            cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s6);
            cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[j % 2][_ni].s7);
          #endif
        }
      }

      // load 到寄存器的编号为 j+1 = 1 ~ 15
      //  最开始 load_stage_idx = 0
      #pragma unroll
      for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
        // apm[(j+1)%2][_mi] = LocalToPrivateA(alm + load_stage_idx * KWG * MWG/VWM, _mi, (j+1));
        #if STRM == 1
          apm[(j+1)%2][_mi] = alm[load_stage_idx * KWG * MWG/VWM + (j+1)*(MWG/VWM) + aLtoP_offset + _mi*MDIMC];
        #elif STRM == 0
          apm[(j+1)%2][_mi] = alm[load_stage_idx * KWG * MWG/VWM + (j+1)*(MWG/VWM) + aLtoP_offset + _mi];
        #endif
      }
      #pragma unroll
      for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
        // bpm[(j+1)%2][_ni] = LocalToPrivateB(blm + load_stage_idx * KWG * NWG/VWN, _ni, (j+1));
        #if STRN == 1
          bpm[(j+1)%2][_ni] = blm[load_stage_idx * KWG * NWG/VWN + (j+1)*(NWG/VWN) + bLtoP_offset + _ni*NDIMC];
        #elif STRN == 0
          bpm[(j+1)%2][_ni] = blm[load_stage_idx * KWG * NWG/VWN + (j+1)*(NWG/VWN) + bLtoP_offset + _ni];
        #endif
      }
      // realM* 不支持进行异或运算
      // alm ^= KWG * MWG/VWM * 4;
      // blm ^= KWG * NWG/VWN * 4;
    }

    // 后续还有块, load 中间寄存器到相应的local memory中
    if (kwg < kSizeK){
      // reg --> Local
      // from a_ldg_reg to alm
      #pragma unroll
      for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
        #pragma unroll
        for (int _kia = 0; _kia < KWA; _kia += 1) {
          // alm[write_stage_idx * KWG * MWG/VWM + (_kia + la1 * KWA)*(MWG/VWM) + la0 + _mia * MDIMA] = a_ldg_reg[_kia * MWA/VWM  + _mia];

          #if STRM == 1
            alm[write_stage_idx * KWG * MWG/VWM + alm_offset + (_kia)*(MWG/VWM) + _mia * MDIMA] = a_ldg_reg[_kia * MWA/VWM  + _mia];
          #elif STRM == 0
            alm[write_stage_idx * KWG * MWG/VWM + alm_offset + (_kia)*(MWG/VWM) + _mia] = a_ldg_reg[_kia * MWA/VWM  + _mia];
          #endif
        }
      }

      // from b_ldg_reg to blm
      #pragma unroll
      for (int _kib = 0; _kib < KWB; _kib += 1) {
        #pragma unroll
        for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {
          // blm[write_stage_idx * KWG * NWG/VWN + (_kib + lb1*KWB)*(NWG/VWN) + lb0 + _nib*NDIMB] = b_ldg_reg[_kib * NWB/VWN + _nib];

          #if STRN == 1
            blm[write_stage_idx * KWG * NWG/VWN + blm_offset + _kib*(NWG/VWN) + _nib*NDIMB] = b_ldg_reg[_kib * NWB/VWN + _nib];
          #elif STRN == 0
            blm[write_stage_idx * KWG * NWG/VWN + blm_offset + _kib*(NWG/VWN) + _nib] = b_ldg_reg[_kib * NWB/VWN + _nib];
          #endif
        }
      }

      // GlobalToLocalA(agm, alm + write_stage_idx * KWG * MWG/VWM, kSizeM, tid, kwg);
      // GlobalToLocalB(bgm, blm + write_stage_idx * KWG * NWG/VWN, kSizeN, tid, kwg);
      
      barrier(CLK_LOCAL_MEM_FENCE);
      write_stage_idx ^= 1;
    }

    // 提前load 下一次循环的数据到reg[0]
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      // apm[0][_mi] = LocalToPrivateA(alm + (load_stage_idx ^ 1) * KWG * MWG/VWM, _mi, 0);
      #if STRM == 1
      apm[0][_mi] = alm[(load_stage_idx ^ 1) * KWG * MWG/VWM + aLtoP_offset + _mi*MDIMC];
      #elif STRM == 0
      apm[0][_mi] = alm[(load_stage_idx ^ 1) * KWG * MWG/VWM + aLtoP_offset + _mi];
      #endif
    }

    #pragma unroll
    for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
      // bpm[0][_ni] = LocalToPrivateB(blm + (load_stage_idx ^ 1) * KWG * NWG/VWN, _ni, 0);
      #if STRN == 1
      bpm[0][_ni] = blm[(load_stage_idx ^ 1) * KWG * NWG/VWN + bLtoP_offset + _ni*NDIMC];
      #elif STRN == 0
      bpm[0][_ni] = blm[(load_stage_idx ^ 1) * KWG * NWG/VWN + bLtoP_offset + _ni];
      #endif
    }

    // 补K 循环中的tail 计算, 由于j取到15还剩下reg[1]没算
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      const realM aval = apm[1][_mi];
      #pragma unroll
      for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
        #if  VWN == 1
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[1][_ni]);
        #elif VWN == 2
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[1][_ni].x);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[1][_ni].y);
        #elif VWN == 4
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[1][_ni].x);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[1][_ni].y);
            cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[1][_ni].z);
            cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[1][_ni].w);
        #elif VWN == 8
            cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s0);
            cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s1);
            cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s2);
            cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s3);
            cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s4);
            cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s5);
            cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s6);
            cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[1][_ni].s7);
        #endif
      }
    }
  } while (kwg < kSizeK);
  
  #if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta 还没检查
  #if GEMMK == 0
    const int cld = kSizeM;
  #endif
  #pragma unroll
  for (int _ni = 0; _ni < NWI; _ni += 1) {
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      StoreResults(cgm, cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);
    }
  }
}

)"
// End of the C++11 raw string literal

// =================================================================================================
