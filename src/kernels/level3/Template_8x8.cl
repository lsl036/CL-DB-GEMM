// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// This is a template of 8x8 micro-kernel of the DB-GEMM kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
INLINE_FUNC void Template_8x8_MicroKernel(  const int NWI, const int MWI, 
                                            const int VWN, const int VWM, 
                                            const realM* apm,
                                            const realN* bpm,
                                            realM* cpm )
    #if GEMMK == 0
    //#pragma unroll
    //for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
        //#pragma unroll
        //for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
            
            realM apm_0 = apm[0];
            realN bpm_0 = bpm[0];
            cpm[0] += apm_0 * bpm_0.x;
            cpm[2] += apm_0 * bpm_0.y; 
            cpm[4] += apm_0 * bpm_0.z;
            cpm[6] += apm_0 * bpm_0.w; 

            realN bpm_1 = bpm[1];
            cpm[8]  += apm_0 * bpm_1.x;
            cpm[10] += apm_0 * bpm_1.y; 
            cpm[12] += apm_0 * bpm_1.z;
            cpm[14] += apm_0 * bpm_1.w; 

            realM apm_1 = apm[1];
            cpm[9]  += apm_1 * bpm_1.x;
            cpm[11] += apm_1 * bpm_1.y; 
            cpm[13] += apm_1 * bpm_1.z;
            cpm[15] += apm_1 * bpm_1.w;

            cpm[1] += apm_1 * bpm_0.x;
            cpm[3] += apm_1 * bpm_0.y; 
            cpm[5] += apm_1 * bpm_0.z;
            cpm[7] += apm_1 * bpm_0.w;           
        //}
    //}
    #endif
)"