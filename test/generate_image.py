"""Image generation examples.

This repo supports two ways to generate an image:
1) Server API (recommended): POST /v1/images/generate with title/description.
   The server generates a 1-line prompt via OpenAI GPT-4.1, then runs Z-Image.
2) Local model run (optional): load Z-Image-Turbo directly.
"""

from __future__ import annotations


def generate_via_api() -> None:
    import os

    import requests

    base_url = (os.getenv("API_BASE_URL") or "http://localhost:8000").rstrip("/")
    payload = {
        "title": "햄릿",
        "description": "덴마크 왕자의 복수와 광기를 다룬 비극. 어두운 궁정, 배신, 유령의 계시.",
    }

    resp = requests.post(f"{base_url}/v1/images/generate", json=payload, timeout=120)
    resp.raise_for_status()

    with open("out.png", "wb") as f:
        f.write(resp.content)
    print("Wrote out.png")


def generate_locally() -> None:
    import torch
    from diffusers import ZImagePipeline

    model_name = "Tongyi-MAI/Z-Image-Turbo"
    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    pipe.transformer.set_attention_backend("flash")
    pipe.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(42)
    image = pipe(
        prompt="Drama movie poster, a lone prince confronting a ghost in a dim palace hall, tense and haunted, cinematic chiaroscuro composition",
        height=1152,
        width=768,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    image.save("generated_image.png")
    print("Wrote generated_image.png")


if __name__ == "__main__":
    # Default to API flow.
    generate_via_api()