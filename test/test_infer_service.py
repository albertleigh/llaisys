"""Tests for the inference service: streaming, concurrent requests, and KV cache pool.

These tests start the FastAPI server in-process and make HTTP requests
to verify end-to-end functionality.

Run with:
    python test/test_infer_service.py --device cpu --model models/DeepSeek-R1-Distill-Qwen-1.5B

Tests:
  1. Non-streaming /v1/chat/completions
  2. Streaming /v1/chat/completions (SSE)
  3. Concurrent requests (multiple prompts, verify independent results)
  4. /v1/models endpoint
  5. /v1/pool/status endpoint (includes KV pool stats)
  6. Request cancellation via client disconnect
"""

import argparse
import asyncio
import json
import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add services/infer to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "infer"))


async def run_tests(model_path: str, device: str):
    """Run all service tests."""
    from httpx import AsyncClient, ASGITransport
    from infer.main import create_app
    from infer.config import Settings

    cfg = Settings(
        model_path=model_path,
        device=device,
        max_ctx_len=512,
        max_batch_size=2,
        max_pool_size=16,
    )

    app = create_app(cfg)

    # Trigger lifespan manually
    from contextlib import asynccontextmanager

    transport = ASGITransport(app=app)

    passed = 0
    failed = 0

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Lifespan events are handled by ASGITransport

        # ── Test 1: Non-streaming completion ──────────────────────────

        print("[Test 1] Non-streaming chat completion...", flush=True)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-r1-distill-qwen-1.5b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                    "stream": False,
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 1.0,
                },
                timeout=120.0,
            )
            assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "choices" in data, f"Missing 'choices' in response: {data}"
            assert len(data["choices"]) > 0, "Empty choices"
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0, "Empty content in response"
            print(f"  Response: {content[:80]}...")
            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

        # ── Test 2: Streaming completion ──────────────────────────────

        print("[Test 2] Streaming chat completion (SSE)...", flush=True)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-r1-distill-qwen-1.5b",
                    "messages": [{"role": "user", "content": "Count to 3"}],
                    "max_tokens": 10,
                    "stream": True,
                    "temperature": 1.0,
                    "top_k": 1,
                    "top_p": 1.0,
                },
                timeout=120.0,
            )
            assert resp.status_code == 200, f"Status {resp.status_code}"

            # Parse SSE events
            chunks = []
            text_content = ""
            for line in resp.text.split("\n"):
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                payload = line.replace("data: ", "").strip() if line.startswith("data:") else line
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    chunks.append(chunk)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        text_content += delta["content"]
                except json.JSONDecodeError:
                    continue

            assert len(chunks) > 0, "No SSE chunks received"
            assert len(text_content) > 0, f"No content in chunks. Got {len(chunks)} chunks."
            print(f"  Received {len(chunks)} chunks, content: {text_content[:80]}...")
            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

        # ── Test 3: Concurrent requests ──────────────────────────────

        print("[Test 3] Concurrent requests...", flush=True)
        try:
            prompts = [
                "What is 1+1?",
                "What color is the sky?",
            ]

            async def make_request(prompt):
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "deepseek-r1-distill-qwen-1.5b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "stream": False,
                        "temperature": 1.0,
                        "top_k": 1,
                        "top_p": 1.0,
                    },
                    timeout=120.0,
                )
                return r

            # Send requests concurrently
            results = await asyncio.gather(*[make_request(p) for p in prompts])

            for i, (prompt, resp) in enumerate(zip(prompts, results)):
                assert resp.status_code == 200, f"Request {i} failed: {resp.status_code}"
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"  Prompt '{prompt}': {content[:60]}...")

            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

        # ── Test 4: /v1/models ────────────────────────────────────────

        print("[Test 4] /v1/models endpoint...", flush=True)
        try:
            resp = await client.get("/v1/models", timeout=10.0)
            assert resp.status_code == 200, f"Status {resp.status_code}"
            data = resp.json()
            assert "data" in data, f"Missing 'data': {data}"
            assert len(data["data"]) > 0, "No models listed"
            model_id = data["data"][0]["id"]
            assert "qwen" in model_id.lower() or "deepseek" in model_id.lower(), \
                f"Unexpected model id: {model_id}"
            print(f"  Model: {model_id}")
            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

        # ── Test 5: /v1/pool/status ──────────────────────────────────

        print("[Test 5] /v1/pool/status endpoint...", flush=True)
        try:
            resp = await client.get("/v1/pool/status", timeout=10.0)
            assert resp.status_code == 200, f"Status {resp.status_code}"
            data = resp.json()

            # Should have pool stats
            assert "pending" in data, f"Missing 'pending': {data}"
            assert "active" in data, f"Missing 'active': {data}"
            assert "max_batch_size" in data, f"Missing 'max_batch_size': {data}"
            assert "loop_running" in data, f"Missing 'loop_running': {data}"

            # Should also have KV pool stats
            assert "kv_pool_max_slots" in data, f"Missing KV pool stats: {data}"
            assert "kv_pool_active" in data, f"Missing KV pool active: {data}"
            assert "kv_pool_free" in data, f"Missing KV pool free: {data}"
            assert "kv_pool_cached_prefixes" in data, f"Missing KV pool cached: {data}"

            print(f"  Pool stats: {json.dumps(data, indent=2)}")
            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

        # ── Test 6: Deterministic output with top_k=1 ────────────────

        print("[Test 6] Deterministic output (top_k=1, same prompt twice)...", flush=True)
        try:
            prompt = "What is the meaning of life?"

            async def deterministic_request():
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "deepseek-r1-distill-qwen-1.5b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "stream": False,
                        "temperature": 1.0,
                        "top_k": 1,
                        "top_p": 1.0,
                    },
                    timeout=120.0,
                )
                return r.json()["choices"][0]["message"]["content"]

            result1 = await deterministic_request()
            result2 = await deterministic_request()

            assert result1 == result2, \
                f"Non-deterministic output!\n  run1: {result1}\n  run2: {result2}"
            print(f"  Both runs: {result1[:80]}...")
            print("  \033[92mPASSED\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[91mFAILED: {e}\033[0m")
            failed += 1

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print(f"\033[91m{failed} test(s) failed!\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference service")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "DeepSeek-R1-Distill-Qwen-1.5B")

    if not os.path.isdir(model_path):
        print(f"Model path not found: {model_path}")
        print("Please provide --model <path>")
        sys.exit(1)

    print(f"Testing inference service (model={model_path}, device={args.device})")
    asyncio.run(run_tests(model_path, args.device))
