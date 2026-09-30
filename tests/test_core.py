"""Smoke tests for the cznn core DSL. Run: python3 tests/test_core.py"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

import cznn  # noqa: E402


def test_seq_forward():
    net = cznn.Seq(cznn.Lin(16, 32), cznn.Lin(32, 8))
    assert net(torch.randn(4, 16)).shape == (4, 8)


def test_res_identity():
    net = cznn.Res(cznn.Lin(16, 16))
    assert net(torch.randn(4, 16)).shape == (4, 16)


def test_add_parallel_branches():
    net = cznn.Add(cznn.Lin(16, 8), cznn.Lin(16, 8))
    assert net(torch.randn(4, 16)).shape == (4, 8)


def test_seq_skips_identity():
    assert isinstance(cznn.Seq(cznn.Identity()), cznn.Identity)


def test_size_counts_parameters():
    net = cznn.Seq(cznn.Lin(16, 32), cznn.Lin(32, 8))
    # Lin() defaults to bias=False (it forwards its third argument to nn.Linear)
    assert cznn.size(net) == 16 * 32 + 32 * 8


def test_neuronloop_fork_imports():
    from cznn import variants_neuronloop as v
    for sym in ("Seq", "Add", "Mul", "Res", "Map", "Einsum", "SwiGLU",
                "Attention", "MultiheadAttention", "Nn", "Weight", "Bias"):
        assert hasattr(v, sym), sym
    assert v.Seq(v.Weight(16, 8))(torch.randn(4, 16)).shape == (16, 8)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
    print("all core tests passed")
