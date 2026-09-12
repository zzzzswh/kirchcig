"""PyTorch autograd wrapper around :class:`kirchcig.KirchhoffCIG`.

Because the operator is linear, the backward pass of ``forward`` is ``adjoint``
and vice versa; each is implemented as an ``autograd.Function`` whose backward
calls the *other* Function, so the graph is recorded during backward as well
and second-order derivatives work.

CUDA tensors with the cuda engine are exchanged with CuPy through DLPack, so
nothing leaves the GPU. Any other combination falls back to a host copy.
"""
from __future__ import annotations

import numpy as np
import torch

from ._operator import KirchhoffCIG

__all__ = ["TorchKirchhoffCIG"]


def _torch_to_cupy(t):
    import cupy as cp
    try:
        return cp.from_dlpack(t)
    except Exception:  # older CuPy / torch
        from torch.utils.dlpack import to_dlpack
        return cp.fromDlpack(to_dlpack(t))


def _cupy_to_torch(a):
    try:
        return torch.from_dlpack(a)
    except Exception:  # older torch
        from torch.utils.dlpack import from_dlpack
        return from_dlpack(a.toDlpack())


def _apply(op: KirchhoffCIG, direction: str, x: torch.Tensor) -> torch.Tensor:
    fn = op.forward if direction == "forward" else op.adjoint
    xd = x.detach()
    if xd.is_cuda and op.engine == "cuda":
        import cupy as cp
        xd = xd.contiguous().to(torch.float32)
        dev = xd.device.index if xd.device.index is not None else torch.cuda.current_device()
        with cp.cuda.Device(dev):
            ptr = torch.cuda.current_stream(xd.device).cuda_stream
            if ptr:  # non-default torch stream: make CuPy launch on it
                with cp.cuda.ExternalStream(ptr):
                    y = fn(_torch_to_cupy(xd))
            else:
                y = fn(_torch_to_cupy(xd))
        return _cupy_to_torch(y)
    xn = np.ascontiguousarray(xd.cpu().numpy(), dtype=np.float32)
    y = np.ascontiguousarray(np.asarray(fn(xn)), dtype=np.float32)
    return torch.from_numpy(y).to(x.device)


class _Forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cig, op):
        ctx.op = op
        return _apply(op, "forward", cig)

    @staticmethod
    def backward(ctx, grad):
        return _Adjoint.apply(grad, ctx.op), None


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, op):
        ctx.op = op
        return _apply(op, "adjoint", data)

    @staticmethod
    def backward(ctx, grad):
        return _Forward.apply(grad, ctx.op), None


class TorchKirchhoffCIG:
    """Differentiable ``forward`` / ``adjoint`` for an existing operator.

    >>> top = TorchKirchhoffCIG(op)
    >>> cig = top.adjoint(data)        # d cig / d data  is  top.forward
    >>> res = top.forward(cig) - data  # d res / d cig   is  top.adjoint
    """

    def __init__(self, op: KirchhoffCIG):
        self.op = op

    def forward(self, cig: torch.Tensor) -> torch.Tensor:
        """Demigration, ``(nh, nx, nz) -> (ns, nr, nt)``."""
        return _Forward.apply(cig, self.op)

    def adjoint(self, data: torch.Tensor) -> torch.Tensor:
        """Migration, ``(ns, nr, nt) -> (nh, nx, nz)``."""
        return _Adjoint.apply(data, self.op)

    __call__ = forward

    @property
    def shape_model(self):
        return self.op.shape_model

    @property
    def shape_data(self):
        return self.op.shape_data

    def __repr__(self):
        return f"TorchKirchhoffCIG({self.op!r})"
