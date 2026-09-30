# cznn — a compositional layer library

**What:** A small, style-driven neural-network composition library. Instead of
subclassing `Module` per architecture, you compose layers point-free:

```python
from cznn import Seq, Add, Res, Lin, size

net = Seq(
    Lin(16, 32),
    Res(Lin(32, 32)),        # residual block
    Lin(32, 8),
)
y = net(x)                   # torch.Tensor
print(size(net))             # parameter count
```

The primitives: `Seq` (flattening composition that skips `Identity`), `Add` /
`Mul` (parallel branches reduced element-wise), `Res` (residual),
`Stack`, `Dot`, `T` (transpose-wrapped submodule), `F` (wrap any callable as a
module), `Lin` / `Sep` / `Left` / `CovLeft` / `ApLeft` / `MpLeft` / `SeLeft`
(layer factories, several of them shape-adaptive `MinModel` choosers), `Decay` /
`Ret` (RetNet-style retention layers), `Bn`, `Lu`, `Stat`, `Log`, plus `size` /
`sized` / `MinModel` for parameter-budget-driven model selection.

**Why it exists:** To run small language-model experiments fast. Composing a
variant takes a few lines, so a whole family of architectures can be compared
under the same training loop — that is why the same file was copied forward
across projects and grew. Measured reuse in the private archive: `from nn import`
appears in **~96 notebooks across four of the author's research projects**
(`cvorat` 38, `cz_bert` 29, `retnet` 15, `neuronloop` 14), plus smaller uses in
`1llm` and others.

**How to run:**

```bash
pip install torch numpy
python3 tests/test_core.py          # 6 tests, no GPU needed
python3 examples/compose_a_model.py # prints shapes and parameter counts
```

## Provenance, stated honestly

This library was extracted from a private archive of personal research code. It
was never a released package; it lived as `nn.py`, copied between projects and
edited in place. The archive contains **three divergent forks**, and they share
only five symbols (`Seq`, `Add`, `Mul`, `Res`, `F`):

| fork | size | distinctive contents | this repo |
|---|---|---|---|
| `retnet/nn.py` | 420 lines | `Decay`, `Ret`, `CovLeft`, `SepCovLeft`, `MpLeft`, `Dat`, `MinModel`, `Stat`, `Log`, `Lu` | **canonical** — `cznn/__init__.py` (verbatim) |
| `neuronloop/nn.py` | 174 lines | `Map`, `Const`, `Weight`, `Bias`, `Cat`, `Einsum`, `SwiGLU`, `Attention`, `MultiheadAttention`, `Nn` | `cznn/variants_neuronloop.py` (verbatim) |
| `cvorat/nn.py` | 201 lines | `Attn`, `rope`, `myrope`, `Plot`, `remap` | not duplicated here — it is published in its own repo (`cvorat`); it also requires `flash_attn` to import |

The forks are kept **verbatim and side by side** rather than silently merged:
merging them is a real refactor with name collisions (`SwiGLU` exists in two
forks with different implementations), and this repo does not pretend that work
was done. The canonical fork is the one that imports cleanly and is the most
complete.

**Status:** archive / research code, extracted 2026-09-30. Not a polished
library: no packaging, no CI, the API is whatever the author needed at the time.
It is published because the reuse evidence is real and it is otherwise only
visible as a side-file inside another project.

**License:** none declared — default copyright applies; this is personal
research code, not a redistribution-ready package.
