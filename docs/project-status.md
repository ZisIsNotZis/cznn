# cznn project status

## Classification

Library — a personal compositional layer library, extracted from a private
research archive (the `nn.py` shared across the author's small-LM projects).

## Status

Archive / research code. Usable and tested (6 core tests pass with torch 2.14,
CPU only), but not packaged and not maintained as a product. The canonical fork
is `retnet/nn.py` (the most complete and the only one that imports without
optional heavy dependencies).

## Evidence

- `cznn/__init__.py` — canonical fork, verbatim from the private archive.
- `cznn/variants_neuronloop.py` — second fork, verbatim (adds
  `Attention` / `MultiheadAttention` / `SwiGLU` / `Einsum` / `Map` / `Nn`).
- `tests/test_core.py` — 6 tests: `Seq` forward, residual identity, parallel
  `Add`, `Identity` elision, parameter counting, and the second fork's import
  surface.
- `examples/compose_a_model.py` — runnable composition demo.
- Reuse measured in the source archive: `from nn import` in ~96 notebooks across
  four projects (`cvorat` 38, `cz_bert` 29, `retnet` 15, `neuronloop` 14).

## Boundaries

- The three forks share only five symbols; they are kept side by side rather
  than merged (name collisions, e.g. two different `SwiGLU`).
- The `cvorat` fork is not duplicated here; it needs `flash_attn` to import.
- `Dat` expects `.map` sidecar files next to a memmapped corpus — it will not run
  without the author's local data.
- No packaging, no CI, no versioning. No license declared (default copyright).
