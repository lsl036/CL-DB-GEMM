## CL-DB-GEMM

This is an optimized, performant and tunable OpenCL double-buffer GEMM repository built based on [CLBlast](https://github.com/CNugteren/CLBlast).
The main work of this repository is implementing a double-buffer GEMM kernel, and utilizing a fine-grained prefetching strategy of the register unit.


### Installation

You can use the same way as CLBlast to copmile and install the kernel.
```
mkdir build && cd build
cmake ..
make
./clblast_sample_sgemm_c
```
We have implemented Double-buffer OpenCL-based Xgemm kernel, as shown in `src/kernels/level3/xgemm_part3.opencl`.


### Tuning with Bayesian Optimization (BO) Tuner

After compiling and installation, we can use BO Tuner for GEMM kernel performance tuning.
```
python RunHKernel.py    // Half precision
python RunSKernel.py    // Single precision
python RunDKernel.py    // Double precision
```
### Cite this work
```
@ARTICLE{11077458,
  author={Lin, Shengle and Xiao, Guoqing and Wang, Haotian and Yang, Wangdong and Li, Kenli and Li, Keqin},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={High Performance OpenCL-Based GEMM Kernel Auto-Tuned by Bayesian Optimization}, 
  year={2025},
  volume={36},
  number={9},
  pages={1985-1997},
  doi={10.1109/TPDS.2025.3587673}}

```
