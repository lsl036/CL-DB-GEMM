// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// This is a template of 4x8 micro-kernel of the DB-GEMM kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
INLINE_FUNC void Template_4x8_MicroKernel(  const int NWI, const int MWI, 
                                            const int VWN, const int VWM, 
                                            const realM* apm,
                                            const realN* bpm,
                                            realM* cpm )
    #if GEMMK == 0
        realM apm_0 = apm[0];
        realN bpm_0 = bpm[0];
        cpm[0] += apm_0 * bpm_0.x;
        cpm[1] += apm_0 * bpm_0.y; 
        cpm[2] += apm_0 * bpm_0.z;
        cpm[3] += apm_0 * bpm_0.w; 

        realN bpm_1 = bpm[1];
        cpm[4] += apm_0 * bpm_1.x;
        cpm[5] += apm_0 * bpm_1.y; 
        cpm[6] += apm_0 * bpm_1.z;
        cpm[7] += apm_0 * bpm_1.w;

    #endif
)"