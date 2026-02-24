"""Unit tests for the Ops.sample operator.

Validates Temperature, Top-K and Top-P (nucleus) sampling against
known invariants and a PyTorch reference implementation.
"""

import sys
import os
from collections import Counter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, zero_tensor, benchmark


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_llaisys_logits(logits_list, dtype_name, device_name):
    """Create a 1-D llaisys tensor from a Python list of floats."""
    t = torch.tensor(logits_list, dtype=_torch_dtype(dtype_name),
                     device=_torch_device(device_name))
    shape = (len(logits_list),)
    lt = llaisys.Tensor(
        shape,
        dtype=_llaisys_dtype(dtype_name),
        device=_llaisys_device(device_name),
    )
    api = llaisys.RuntimeAPI(_llaisys_device(device_name))
    api.memcpy_sync(
        lt.data_ptr(),
        t.data_ptr(),
        t.numel() * t.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return lt


def _torch_dtype(name):
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[name]


def _torch_device(name):
    if name == "cpu":
        return torch.device("cpu")
    elif name == "nvidia":
        return torch.device("cuda:0")
    raise ValueError(f"Unknown device: {name}")


def _llaisys_dtype(name):
    return {"f32": llaisys.DataType.F32, "f16": llaisys.DataType.F16, "bf16": llaisys.DataType.BF16}[name]


def _llaisys_device(name):
    m = {"cpu": llaisys.DeviceType.CPU}
    if hasattr(llaisys.DeviceType, "NVIDIA"):
        m["nvidia"] = llaisys.DeviceType.NVIDIA
    return m[name]


# ── Torch reference implementation ───────────────────────────────────────────

def torch_sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Torch-based sampling matching the llaisys semantics."""
    scores = logits.float()

    # Greedy
    if temperature <= 0.0 or top_k == 1:
        return int(scores.argmax().item())

    # Temperature
    scores = scores / temperature

    # Top-K
    if top_k > 0 and top_k < scores.numel():
        kth = torch.topk(scores, top_k).values[-1]
        scores[scores < kth] = float("-inf")

    # Softmax
    probs = torch.softmax(scores, dim=-1)

    # Top-P
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

    return int(torch.multinomial(probs, 1).item())


# ── Test 1: Greedy (temperature <= 0 OR top_k == 1) always picks argmax ─────

def test_greedy(dtype_name="f32", device_name="cpu"):
    """With temperature=0 or top_k=1, sample must be identical to argmax."""
    print(f"  test_greedy  dtype <{dtype_name}>")
    logits = [0.1, 0.5, 0.9, 0.3, 0.7]
    lt = _make_llaisys_logits(logits, dtype_name, device_name)

    expected = logits.index(max(logits))  # index 2

    # temperature <= 0 → greedy
    for _ in range(20):
        idx = llaisys.Ops.sample(lt, temperature=0.0, top_k=0, top_p=1.0)
        assert idx == expected, f"greedy (temp=0) expected {expected}, got {idx}"

    # top_k == 1 → greedy regardless of temperature
    for _ in range(20):
        idx = llaisys.Ops.sample(lt, temperature=1.0, top_k=1, top_p=1.0)
        assert idx == expected, f"greedy (top_k=1) expected {expected}, got {idx}"


# ── Test 2: Output is always a valid index ───────────────────────────────────

def test_valid_index(dtype_name="f32", device_name="cpu"):
    """Sampled index must be in [0, vocab_size) for every config."""
    print(f"  test_valid_index  dtype <{dtype_name}>")
    vocab_size = 128
    _, lt = random_tensor((vocab_size,), dtype_name, device_name)

    configs = [
        {"temperature": 0.5, "top_k": 0, "top_p": 1.0},
        {"temperature": 1.0, "top_k": 10, "top_p": 1.0},
        {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
        {"temperature": 1.5, "top_k": 20, "top_p": 0.5},
        {"temperature": 0.0, "top_k": 0, "top_p": 1.0},  # greedy
    ]
    for cfg in configs:
        for _ in range(50):
            idx = llaisys.Ops.sample(lt, **cfg)
            assert 0 <= idx < vocab_size, (
                f"index {idx} out of range for vocab_size={vocab_size}, cfg={cfg}"
            )


# ── Test 3: Top-K restricts sampling to K highest logits ────────────────────

def test_top_k(dtype_name="f32", device_name="cpu"):
    """With top_k=2, only the two highest-logit indices should appear."""
    print(f"  test_top_k  dtype <{dtype_name}>")
    # Indices 3 and 1 are the top-2
    logits = [0.1, 0.8, 0.2, 0.9, 0.05]
    lt = _make_llaisys_logits(logits, dtype_name, device_name)

    top2 = {1, 3}
    for _ in range(200):
        idx = llaisys.Ops.sample(lt, temperature=1.0, top_k=2, top_p=1.0)
        assert idx in top2, f"top_k=2: got index {idx}, expected one of {top2}"


# ── Test 4: Top-P restricts sampling to nucleus ─────────────────────────────

def test_top_p(dtype_name="f32", device_name="cpu"):
    """With top_p=0.5, only indices whose cumulative probability < 0.5 should
    appear (plus the one that crosses the threshold)."""
    print(f"  test_top_p  dtype <{dtype_name}>")
    # Make logits with a clear dominant token.
    # After softmax the highest will grab most of the mass.
    logits = [-10.0, -10.0, 10.0, -10.0, -10.0]
    lt = _make_llaisys_logits(logits, dtype_name, device_name)

    # Index 2 should dominate with top_p = 0.5
    for _ in range(200):
        idx = llaisys.Ops.sample(lt, temperature=1.0, top_k=0, top_p=0.5)
        assert idx == 2, f"top_p=0.5 with dominant logit: got {idx}, expected 2"


# ── Test 5: Temperature affects distribution spread ──────────────────────────

def test_temperature(dtype_name="f32", device_name="cpu"):
    """Low temperature should concentrate samples on argmax; high temperature
    should spread them out."""
    print(f"  test_temperature  dtype <{dtype_name}>")
    logits = [0.0, 1.0, 2.0, 0.5, 0.3]
    lt = _make_llaisys_logits(logits, dtype_name, device_name)

    n_samples = 500

    # Low temperature → nearly all samples at argmax (index 2)
    low_counts = Counter()
    for _ in range(n_samples):
        low_counts[llaisys.Ops.sample(lt, temperature=0.01, top_k=0, top_p=1.0)] += 1
    assert low_counts[2] > n_samples * 0.95, (
        f"low temp: argmax count {low_counts[2]}/{n_samples} too low"
    )

    # High temperature → more diversity (argmax should be less dominant)
    high_counts = Counter()
    for _ in range(n_samples):
        high_counts[llaisys.Ops.sample(lt, temperature=5.0, top_k=0, top_p=1.0)] += 1
    unique_high = len(high_counts)
    assert unique_high >= 3, (
        f"high temp: only {unique_high} unique tokens, expected ≥ 3"
    )


# ── Test 6: Large vocab ─────────────────────────────────────────────────────

def test_large_vocab(dtype_name="f32", device_name="cpu"):
    """Sample should work correctly on vocab-sized tensors (e.g. 151936)."""
    print(f"  test_large_vocab  dtype <{dtype_name}>")
    vocab_size = 151936  # Qwen2 vocab size
    _, lt = random_tensor((vocab_size,), dtype_name, device_name)

    for _ in range(10):
        idx = llaisys.Ops.sample(lt, temperature=0.8, top_k=50, top_p=0.9)
        assert 0 <= idx < vocab_size, f"index {idx} out of range"

    # Greedy should also work
    idx1 = llaisys.Ops.sample(lt, temperature=0.0, top_k=0, top_p=1.0)
    idx2 = llaisys.Ops.sample(lt, temperature=0.0, top_k=0, top_p=1.0)
    assert idx1 == idx2, "greedy on same logits should be deterministic"


# ── Test 7: Benchmark ────────────────────────────────────────────────────────

def test_benchmark(shape, dtype_name="f32", device_name="cpu",
                   temperature=0.8, top_k=50, top_p=0.9):
    """Benchmark llaisys.Ops.sample vs torch equivalent."""
    print(f"   shape {shape} dtype <{dtype_name}>  "
          f"temp={temperature} top_k={top_k} top_p={top_p}")
    torch_logits, llaisys_logits = random_tensor(shape, dtype_name, device_name)

    benchmark(
        lambda: torch_sample(torch_logits, temperature, top_k, top_p),
        lambda: llaisys.Ops.sample(llaisys_logits,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p),
        device_name,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_dtypes = ["f32", "f16", "bf16"]

    print(f"Testing Ops.sample on {args.device}")
    for dtype_name in test_dtypes:
        test_greedy(dtype_name, args.device)
    for dtype_name in test_dtypes:
        test_valid_index(dtype_name, args.device)
    for dtype_name in test_dtypes:
        test_top_k(dtype_name, args.device)
    for dtype_name in test_dtypes:
        test_top_p(dtype_name, args.device)
    for dtype_name in test_dtypes:
        test_temperature(dtype_name, args.device)
    for dtype_name in test_dtypes:
        test_large_vocab(dtype_name, args.device)

    if args.profile:
        print(f"\nBenchmarking Ops.sample on {args.device}")
        test_shapes = [(4096,), (151936,)]
        configs = [
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0},    # greedy
            {"temperature": 0.8, "top_k": 50, "top_p": 0.9},   # typical sampling
            {"temperature": 1.0, "top_k": 0, "top_p": 0.9},    # top-p only
        ]
        for shape in test_shapes:
            for cfg in configs:
                for dtype_name in test_dtypes:
                    test_benchmark(shape, dtype_name, args.device, **cfg)

    print("\033[92mAll sample tests passed!\033[0m\n")
