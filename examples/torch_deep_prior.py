"""Deep-prior least-squares migration in a few lines.

The CIG is parametrised by a small network; autograd differentiates through
the demigration operator, whose backward pass is the migration operator.
Needs torch (and CuPy + a GPU for engine="cuda", where tensors never leave
the device).
"""
import torch

from kirchcig import KirchhoffCIG
from kirchcig.torch import TorchKirchhoffCIG

op = KirchhoffCIG.demo(engine="auto", nh=4, hmax=500.0)
top = TorchKirchhoffCIG(op)
device = "cuda" if (op.engine == "cuda" and torch.cuda.is_available()) else "cpu"

data = torch.from_numpy(op.demo_data()).to(device)
nh, nx, nz = op.shape_model

net = torch.nn.Sequential(
    torch.nn.Conv2d(nh, 16, 3, padding=1), torch.nn.GELU(),
    torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.GELU(),
    torch.nn.Conv2d(16, nh, 3, padding=1),
).to(device)
latent = torch.randn(1, nh, nx, nz, device=device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for it in range(200):
    cig = net(latent)[0]                              # (nh, nx, nz)
    loss = 0.5 * (top.forward(cig) - data).pow(2).sum()
    opt.zero_grad()
    loss.backward()                                   # backward of forward == op.adjoint
    opt.step()
    if it % 50 == 0 or it == 199:
        print(f"iter {it:4d}  data misfit {loss.item():.4e}")

img = net(latent)[0].detach().sum(0).cpu().numpy()
print("done; stacked image shape", img.shape)
