# CPU emulation of the CUDA kernels

Development tool, not part of the installed package. Needs `g++` and pthreads.

`emu.cpp` includes the kernel source verbatim (written to `kernel_body.inc` by
`fake_cupy.py`) with CUDA keywords stubbed out, runs each block's threads as
real `std::thread`s with a generation barrier for `__syncthreads`, and
implements `atomicCAS`/`atomicAdd` with GCC atomics. `fake_cupy.py` installs a
numpy-backed `cupy` module whose `RawKernel` compiles that emulation with the
same `-D` flags CuPy would pass and launches it with the same grid.

```
python dev/cuda_emulation/run_emulated_tests.py
```

Every cuda-marked test then runs against the numpy oracle. With float64
accumulation the emulated kernels reproduce the numpy engine bit-for-bit.
