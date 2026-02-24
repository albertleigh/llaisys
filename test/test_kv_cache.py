"""Tests for KV cache management C API, Python bindings, and KV cache pool.

Run with:
    python test/test_kv_cache.py --device cpu --model models/DeepSeek-R1-Distill-Qwen-1.5B

Tests:
  1. KV cache save/restore snapshot
  2. KV cache reset
  3. Prefix matching in the KV cache pool
  4. Multi-request KV swap correctness (two prompts produce same tokens
     whether run sequentially with reset or with KV swap)
  5. Prefix match produces identical continuation
"""

import gc
from test_utils import *

import argparse
from transformers import AutoTokenizer
import torch
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_model_and_tokenizer(model_path, device_name="cpu", max_ctx_len=2048):
    """Load tokenizer and llaisys model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name), max_ctx_len=max_ctx_len)
    return tokenizer, model


def tokenize_prompt(tokenizer, prompt):
    """Apply chat template and tokenize a prompt."""
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(input_content)


def generate_tokens(model, input_ids, max_steps=5):
    """Generate tokens using the model's infer_step."""
    tokens = list(input_ids)
    next_input = list(input_ids)
    generated = []
    for _ in range(max_steps):
        token_id = model.infer_step(next_input)
        generated.append(token_id)
        tokens.append(token_id)
        if token_id == model.meta.end_token:
            break
        next_input = [token_id]
    return generated


# ── Test 1: KV cache save/restore yields same results ───────────────

def test_kv_save_restore(model, tokenizer):
    """Test that saving and restoring KV state produces identical output."""
    print("[Test 1] KV cache save/restore (host-memory snapshot)...", flush=True)

    prompt = "What is 2+2?"
    input_ids = tokenize_prompt(tokenizer, prompt)

    # Run 1: Generate normally
    model.reset_kv_cache()
    tokens_run1 = generate_tokens(model, input_ids, max_steps=10)

    # Run 2: Generate with save/restore mid-way
    model.reset_kv_cache()

    # First, process the prompt
    first_token = model.infer_step(input_ids)

    # Save KV state after prompt processing — copies to CPU
    saved_snapshot = model.save_kv_state()
    saved_pos = model.get_pos()

    # Verify snapshot stores the right position
    snap_pos = model.get_snapshot_pos(saved_snapshot)
    assert snap_pos == saved_pos, \
        f"Snapshot pos={snap_pos} != model pos={saved_pos}"

    # Continue generating a few tokens
    remaining1 = [first_token]
    gen2 = [first_token]
    for _ in range(4):
        t = model.infer_step(remaining1)
        gen2.append(t)
        if t == model.meta.end_token:
            break
        remaining1 = [t]

    # Reset device KV cache (simulates freeing device memory)
    model.reset_kv_cache()

    # Restore the saved snapshot (CPU → device)
    model.restore_kv_state(saved_snapshot)
    assert model.get_pos() == saved_pos, \
        f"Position mismatch after restore: {model.get_pos()} != {saved_pos}"

    # Re-generate from the saved state — should produce same tokens
    remaining2 = [first_token]
    gen3 = [first_token]
    for _ in range(4):
        t = model.infer_step(remaining2)
        gen3.append(t)
        if t == model.meta.end_token:
            break
        remaining2 = [t]

    assert gen2 == gen3, \
        f"Tokens differ after restore!\n  run2: {gen2}\n  run3: {gen3}"

    # Free the snapshot
    model.free_kv_snapshot(saved_snapshot)

    print("  \033[92mPASSED\033[0m")


# ── Test 2: KV cache reset ──────────────────────────────────────────

def test_kv_reset(model, tokenizer):
    """Test that resetting KV cache produces same results as fresh model."""
    print("[Test 2] KV cache reset...", flush=True)

    prompt = "Hello!"
    input_ids = tokenize_prompt(tokenizer, prompt)

    # Run 1
    model.reset_kv_cache()
    tokens_run1 = generate_tokens(model, input_ids, max_steps=5)

    # Pollute the KV cache with a different prompt
    model.reset_kv_cache()
    other_ids = tokenize_prompt(tokenizer, "What is machine learning?")
    generate_tokens(model, other_ids, max_steps=3)

    # Reset and run same prompt again
    model.reset_kv_cache()
    tokens_run2 = generate_tokens(model, input_ids, max_steps=5)

    assert tokens_run1 == tokens_run2, \
        f"Tokens differ after reset!\n  run1: {tokens_run1}\n  run2: {tokens_run2}"

    print("  \033[92mPASSED\033[0m")


# ── Test 3: KV cache pool prefix matching ────────────────────────────

def test_kv_pool_prefix_matching():
    """Test the KVCachePool prefix matching logic (no model needed)."""
    print("[Test 3] KV cache pool prefix matching...", flush=True)

    # Import the pool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "infer"))
    from infer.engine.kv_cache_pool import KVCachePool, KVSlot, _common_prefix_length

    # Test _common_prefix_length
    assert _common_prefix_length([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _common_prefix_length([1, 2, 3, 4], [1, 2, 3]) == 3
    assert _common_prefix_length([1, 2, 3], [1, 2, 3]) == 3
    assert _common_prefix_length([1, 2, 3], [4, 5, 6]) == 0
    assert _common_prefix_length([], [1, 2, 3]) == 0
    assert _common_prefix_length([1], [1]) == 1

    # Test pool operations
    pool = KVCachePool(max_slots=4)

    # Acquire slot for request A
    slot_a, prefix_a = pool.acquire_slot("req_a", [1, 2, 3, 4, 5])
    assert prefix_a == 0, "First request should have no prefix match"
    assert slot_a.request_id == "req_a"

    # Simulate saving state (snapshot is an opaque handle; use a dummy value)
    pool.update_slot(slot_a, [1, 2, 3, 4, 5], "dummy_snapshot", pos=5)
    pool.release_slot(slot_a)
    assert slot_a.is_free

    # Request B with same prefix should match
    slot_b, prefix_b = pool.acquire_slot("req_b", [1, 2, 3, 4, 5, 6, 7])
    assert prefix_b == 5, f"Expected prefix match of 5, got {prefix_b}"
    assert slot_b.slot_id == slot_a.slot_id, "Should reuse same slot"
    pool.release_slot(slot_b)

    # Request C with different prefix should not match
    slot_c, prefix_c = pool.acquire_slot("req_c", [10, 20, 30])
    assert prefix_c == 0, f"Expected no prefix match, got {prefix_c}"
    assert slot_c.slot_id != slot_a.slot_id or slot_c.kv_snapshot is None
    pool.release_slot(slot_c)

    # Test pool stats
    stats = pool.stats()
    assert stats["kv_pool_max_slots"] == 4
    assert stats["kv_pool_active"] == 0

    # Test exhaustion
    slots = []
    for i in range(4):
        s, _ = pool.acquire_slot(f"req_{i}", [i])
        slots.append(s)

    try:
        pool.acquire_slot("req_overflow", [99])
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    for s in slots:
        pool.release_slot(s)

    print("  \033[92mPASSED\033[0m")


# ── Test 4: Multi-request KV swap correctness ───────────────────────

def test_multi_request_kv_swap(model, tokenizer):
    """Test that swapping KV caches between requests produces correct results.

    Generate tokens for two different prompts. First run them independently
    (with reset between), then run them interleaved (with KV swap between):
    - Process prompt A, save KV state
    - Process prompt B, save KV state
    - Restore A's KV state, generate 1 more token
    - Restore B's KV state, generate 1 more token
    - Compare against the independent run.
    """
    print("[Test 4] Multi-request KV swap correctness...", flush=True)

    prompt_a = "What is the capital of France?"
    prompt_b = "Explain gravity."

    ids_a = tokenize_prompt(tokenizer, prompt_a)
    ids_b = tokenize_prompt(tokenizer, prompt_b)

    # Independent runs
    model.reset_kv_cache()
    tokens_a_independent = generate_tokens(model, ids_a, max_steps=5)

    model.reset_kv_cache()
    tokens_b_independent = generate_tokens(model, ids_b, max_steps=5)

    # Interleaved runs with KV swap (only 1 device KV cache at a time)
    # Process prompt A
    model.reset_kv_cache()
    first_a = model.infer_step(ids_a)
    state_a = model.save_kv_state()
    tokens_a_swapped = [first_a]

    # Reset device KV cache before processing B
    model.reset_kv_cache()

    # Process prompt B
    first_b = model.infer_step(ids_b)
    state_b = model.save_kv_state()
    tokens_b_swapped = [first_b]

    # Alternate generating tokens
    for step in range(4):
        # Generate one token for A — restore from CPU snapshot
        model.reset_kv_cache()
        model.restore_kv_state(state_a)
        next_a = model.infer_step([tokens_a_swapped[-1]])
        tokens_a_swapped.append(next_a)
        # Free old snapshot, save new one
        model.free_kv_snapshot(state_a)
        state_a = model.save_kv_state()

        # Generate one token for B — restore from CPU snapshot
        model.reset_kv_cache()
        model.restore_kv_state(state_b)
        next_b = model.infer_step([tokens_b_swapped[-1]])
        tokens_b_swapped.append(next_b)
        model.free_kv_snapshot(state_b)
        state_b = model.save_kv_state()

    # Free final snapshots
    model.free_kv_snapshot(state_a)
    model.free_kv_snapshot(state_b)

    assert tokens_a_independent == tokens_a_swapped, \
        f"Prompt A tokens differ!\n  independent: {tokens_a_independent}\n  swapped: {tokens_a_swapped}"
    assert tokens_b_independent == tokens_b_swapped, \
        f"Prompt B tokens differ!\n  independent: {tokens_b_independent}\n  swapped: {tokens_b_swapped}"

    print("  \033[92mPASSED\033[0m")


# ── Test 5: Prefix match produces identical continuation ─────────────

def test_prefix_match_correctness(model, tokenizer):
    """Test that using prefix matching (reusing KV cache from a shorter
    prompt) produces the exact same tokens as processing the full prompt.

    Simulates what the inference engine does with prefix matching:
    1. Process short prompt → save KV state
    2. New longer prompt that starts with the same tokens
    3. Restore KV state, process only the new tokens
    4. Compare generated output against processing the full prompt from scratch
    """
    print("[Test 5] Prefix match correctness...", flush=True)

    # Create two prompts where the second extends the first
    prompt_base = "Tell me about"
    prompt_extended = "Tell me about machine learning"

    ids_base = tokenize_prompt(tokenizer, prompt_base)
    ids_extended = tokenize_prompt(tokenizer, prompt_extended)

    # Find actual common prefix length
    prefix_len = 0
    for i in range(min(len(ids_base), len(ids_extended))):
        if ids_base[i] == ids_extended[i]:
            prefix_len = i + 1
        else:
            break

    print(f"  Base prompt: {len(ids_base)} tokens, Extended: {len(ids_extended)} tokens, Common prefix: {prefix_len}")

    if prefix_len < 2:
        print("  Skipping — prompts don't share a useful prefix")
        print("  \033[93mSKIPPED\033[0m")
        return

    # Run 1: Process full extended prompt from scratch
    model.reset_kv_cache()
    tokens_full = generate_tokens(model, ids_extended, max_steps=5)

    # Run 2: Process base prompt, save state, then continue with only new tokens
    model.reset_kv_cache()
    # Process the common prefix
    _ = model.infer_step(ids_extended[:prefix_len])
    saved_snapshot = model.save_kv_state()

    # Reset device memory, then restore snapshot
    model.reset_kv_cache()
    model.restore_kv_state(saved_snapshot)
    remaining = ids_extended[prefix_len:]
    if len(remaining) > 0:
        first_continuation = model.infer_step(remaining)
    else:
        first_continuation = model.infer_step([ids_extended[-1]])
        # Need to adjust — this is a degenerate case

    tokens_prefix = [first_continuation]
    next_input = [first_continuation]
    for _ in range(4):
        t = model.infer_step(next_input)
        tokens_prefix.append(t)
        if t == model.meta.end_token:
            break
        next_input = [t]

    assert tokens_full == tokens_prefix, \
        f"Prefix-matched tokens differ from full processing!\n  full:   {tokens_full}\n  prefix: {tokens_prefix}"

    # Free snapshot
    model.free_kv_snapshot(saved_snapshot)

    print("  \033[92mPASSED\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KV cache management")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        # Default model path
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "DeepSeek-R1-Distill-Qwen-1.5B")

    if not os.path.isdir(model_path):
        print(f"Model path not found: {model_path}")
        print("Please provide --model <path>")
        sys.exit(1)

    print(f"Loading model from {model_path} (device={args.device})")
    tokenizer, model = load_model_and_tokenizer(model_path, args.device)
    print(f"Model loaded: {model.meta.nlayer} layers, vocab={model.meta.voc}")

    # Run all tests
    passed = 0
    failed = 0
    total = 5

    tests = [
        ("test_kv_save_restore", lambda: test_kv_save_restore(model, tokenizer)),
        ("test_kv_reset", lambda: test_kv_reset(model, tokenizer)),
        ("test_kv_pool_prefix_matching", lambda: test_kv_pool_prefix_matching()),
        ("test_multi_request_kv_swap", lambda: test_multi_request_kv_swap(model, tokenizer)),
        ("test_prefix_match_correctness", lambda: test_prefix_match_correctness(model, tokenizer)),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  \033[91mFAILED: {e}\033[0m")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print(f"\033[91m{failed} test(s) failed!\033[0m")
        sys.exit(1)
