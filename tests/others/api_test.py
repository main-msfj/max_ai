import time, asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
    api_key="ollama",
)

async def bench(model, thinking, prompt="What is 15% of 240? Show your reasoning."):
    suffix = "" if thinking else " /no_think"
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + suffix}],
    )
    dt = time.perf_counter() - t0
    tok = r.usage.completion_tokens
    content = r.choices[0].message.content or ""
    print(f"{model:15s} think={thinking!s:5s} | {dt:5.2f}s | {tok:5d} tok | {tok/dt:5.1f} tok/s")
    print(f"   -> {content[:120]!r}")
    print()

async def main():
    # Warm up
    await client.chat.completions.create(model="qwen3:1.7b", messages=[{"role":"user","content":"hi /no_think"}], max_tokens=5)
    
    await bench("qwen3:1.7b", False)
    await bench("qwen3:1.7b", True)
    await bench("qwen3:4b", False)
    await bench("qwen3:4b", True)

asyncio.run(main())