"""Run the package's cuda-marked tests without a GPU.

The real CUDA source in kirchcig/_kernels.py is compiled with g++ into a CPU
emulation (std::thread per CUDA thread, barrier for __syncthreads, CAS
atomics) and a numpy-backed stand-in for cupy routes RawKernel launches to it.
This validates kernel logic and the launch configuration in _engine_cuda.py;
it says nothing about NVRTC compilation or performance.

    python dev/cuda_emulation/run_emulated_tests.py
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
conftest = os.path.join(root, "tests", "conftest.py")
if os.path.exists(conftest):
    raise SystemExit("tests/conftest.py already exists; refusing to overwrite")
with open(conftest, "w") as f:
    f.write(f"import sys\nsys.path.insert(0, {here!r})\nimport fake_cupy\nfake_cupy.install()\n")
try:
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests", "-k", "cuda"], cwd=root))
finally:
    os.remove(conftest)
