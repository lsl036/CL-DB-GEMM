// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// This is a template of 8x4 micro-kernel of the DB-GEMM kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
INLINE_FUNC void Template_8x4_MicroKernel(  const int NWI, const int MWI, 
                                            const int VWN, const int VWM, 
                                            const realM* apm,
                                            const realN* bpm,
                                            realM* cpm )
    #if GEMMK == 0
        realM apm_0 = apm[0];
        realN bpm_0 = bpm[0];
        cpm[0] += apm_0 * bpm_0.x;
        cpm[2] += apm_0 * bpm_0.y; 
        cpm[4] += apm_0 * bpm_0.z;
        cpm[6] += apm_0 * bpm_0.w; 

        realM apm_1 = apm[1];
        cpm[1] += apm_1 * bpm_0.x;
        cpm[3] += apm_1 * bpm_0.y; 
        cpm[5] += apm_1 * bpm_0.z;
        cpm[7] += apm_1 * bpm_0.w;

    #endif
)"