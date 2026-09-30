"""Compose a small model with the cznn DSL and show shapes / parameter counts."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cznn import Add, Lin, MinModel, Res, Seq, size  # noqa: E402

x = torch.randn(4, 16)

plain = Seq(Lin(16, 32), Lin(32, 8), Lin(8, 4))
residual = Seq(Lin(16, 32), Res(Lin(32, 32)), Lin(32, 4))
branches = Add(Lin(16, 4), Lin(16, 4))

for name, net in [("plain", plain), ("residual", residual), ("parallel-Add", branches)]:
    print(f"{name:14s} out={tuple(net(x).shape)}  params={size(net)}")

# MinModel picks the cheapest candidate that satisfies the caller's factory call
wider = Seq(Lin(16, 4))
narrower = Seq(Lin(16, 2))
print("MinModel picks the cheaper:", size(MinModel(wider, narrower)))
