"""A numpy-backed stand-in for cupy that routes RawKernel launches to the CPU
emulation of the real CUDA source. Only what kirchcig touches is implemented."""
import ctypes, hashlib, os, subprocess, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SMEM_OPTIN = 96 * 1024

class _Runtime:
    @staticmethod
    def getDeviceCount(): return 1

class _Device:
    id = 0
    attributes = {"MultiProcessorCount": 4, "MaxSharedMemoryPerBlockOptin": SMEM_OPTIN}
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False

class RawKernel:
    def __init__(self, code, name, options=(), backend="nvrtc"):
        self.code, self.name, self.options = code, name, options
        self.max_dynamic_shared_size_bytes = 48 * 1024
        self._lib = None
    def compile(self):
        macros = dict(o[2:].split("=") for o in self.options)
        self.macros = macros
        key = hashlib.md5((self.code + "|".join(self.options)).encode()).hexdigest()[:12]
        so = os.path.join(HERE, f"lib_fake_{key}.so")
        if not os.path.exists(so):
            open(os.path.join(HERE, "kernel_body.inc"), "w").write(self.code)
            cmd = ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-DTCHUNK_MAX=65536",
                   *self.options, os.path.join(HERE, "emu.cpp"), "-o", so]
            subprocess.check_call(cmd)
        self._lib = ctypes.CDLL(so)
    def __call__(self, grid, block, args, shared_mem=0):
        if self._lib is None: self.compile()
        m = self.macros
        acc_bytes = 8 if m["ACC"] == "double" else 4
        # what CuPy would reject: too much dynamic smem without opt-in
        limit = 48 * 1024 if shared_mem <= 48 * 1024 else self.max_dynamic_shared_size_bytes
        assert shared_mem <= limit <= SMEM_OPTIN, ("shared mem", shared_mem, limit)
        if self.name == "kirch_adjoint":
            assert shared_mem == int(m["NH"]) * int(m["BLOCK"]) * acc_bytes, "adjoint smem"
        else:
            facc_bytes = 8 if int(m.get("AA", "0")) else acc_bytes
            tchunk = int(args[10]); assert shared_mem == tchunk * facc_bytes, "forward smem"
        # the data-side buffers must have the dtype the kernel was compiled for
        aa = int(m.get("AA", "0"))
        if self.name == "kirch_adjoint":
            in_dt = np.float64 if aa else np.float32
            out_dt = np.float32 if m["OUT"] == "float" else np.float64
        else:
            in_dt = np.float32
            out_dt = np.float64 if aa else np.float32
        assert args[0].dtype == in_dt, ("input dtype", args[0].dtype, in_dt)
        assert args[5].dtype == out_dt, ("output dtype", args[5].dtype, out_dt)
        cargs = []
        for a in args:
            if isinstance(a, np.ndarray):
                assert a.flags.c_contiguous
                cargs.append(a.ctypes.data_as(ctypes.c_void_p))
            elif isinstance(a, np.int32): cargs.append(ctypes.c_int(int(a)))
            elif isinstance(a, np.float32): cargs.append(ctypes.c_float(float(a)))
            else: raise TypeError(f"unsupported kernel arg {type(a)}")
        fn = getattr(self._lib, "launch_" + self.name)
        fn(ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(block[0]), *cargs)

def install():
    cp = types.ModuleType("cupy")
    cp.ndarray = np.ndarray
    cp.asarray = lambda x, dtype=None: np.asarray(x, dtype=dtype)
    cp.ascontiguousarray = np.ascontiguousarray
    cp.empty = np.empty
    cp.zeros = np.zeros
    cp.cumsum = np.cumsum
    cp.float32, cp.float64 = np.float32, np.float64
    cp.RawKernel = RawKernel
    cp.cuda = types.SimpleNamespace(Device=_Device, runtime=_Runtime)
    sys.modules["cupy"] = cp
    return cp
