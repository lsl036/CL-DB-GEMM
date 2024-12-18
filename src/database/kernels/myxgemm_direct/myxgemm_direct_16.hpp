
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xgemm_Direct16' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry MyXgemmDirectHalf = {
  "MyXgemmDirect", Precision::kHalf, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "}, Params{ 8, 32, 8, 8, 32, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 8, 32, 8, 8, 32, 1, 1, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "default", {
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 4, 64, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 2, 4, 64, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T628                                         "}, Params{ 2, 16, 16, 8, 8, 1, 1, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Mali-T760                                         "}, Params{ 2, 16, 16, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 8, 8, 1, 1, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) HD Graphics 620                          "}, Params{ 2, 16, 16, 16, 16, 1, 1, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "}, Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 2, 8, 8, 8, 8, 1, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Jingjia Micro GPUs
      kDeviceTypeGPU, "Jingjia Micro", {
        { "default", {
          { Name{"Jingjia OpenCL Device JM9200.6304.0130            "}, Params{ 16, 8, 8, 8, 32, 1, 0, 8, 2, 64, 0, 0, 0, 0, 0, 0 } },
          { Name{"Jingjia OpenCL Device JM9230.6304.0130            "}, Params{ 8, 16, 16, 8, 16, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 16, 8, 8, 8, 32, 1, 0, 8, 2, 64, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "default", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 16, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 16, 16, 16, 8, 8, 1, 1, 2, 4, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                        , Params{ 2, 16, 16, 16, 16, 1, 1, 1, 1, 16, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
