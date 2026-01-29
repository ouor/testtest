from __future__ import annotations

from typing import Any

import gradio as gr

from demo.code_examples.generators import ExampleRequest, generate_all
from demo.components.code_panel import CodePanel
from demo.components.r2_toggle import r2_toggle
from demo.components.result_viewers import safe_json_or_text
from demo.components.two_column import two_column
from demo.http_client import HttpClient
from demo.tabs.image_generation.api import post_generate_image_png, post_generate_image_to_r2
from demo.utils import join_api, write_temp_bytes


def build_image_generation_tab(*, base_url: gr.Textbox, timeout: gr.Number) -> None:
    left, right = two_column(left_title="Demo", right_title="Example code")

    with left:
        use_r2 = r2_toggle(default=False)
        prompt = gr.Textbox(label="Prompt", value="A serene landscape with mountains and a river during sunset.")
        seed = gr.Number(label="Seed", value=42, precision=0)
        r2_key = gr.Textbox(label="(R2) key (optional)", placeholder="images/generated/custom.png", visible=False)
        run = gr.Button("Generate")
        status = gr.Markdown("")
        image_out = gr.Image(label="Generated image", type="filepath", visible=True)
        json_out = gr.JSON(label="Response", visible=False)

    init_req = ExampleRequest(
        method="POST",
        url=join_api(base_url.value or "http://localhost:8000", "/v1/images/generate"),
        json_body={"prompt": "A serene landscape with mountains and a river during sunset.", "seed": 42},
    )
    init_examples = generate_all(init_req)

    with right:
        panel = CodePanel().build(initial=init_examples)

    def _example(api_base_url: str, r2: bool, p: str, s: float, key: str):
        payload: dict[str, Any] = {"prompt": p, "seed": int(s) if s is not None else 42}
        if r2 and key.strip():
            payload["key"] = key.strip()

        path = "/v1/r2/images/generate" if r2 else "/v1/images/generate"
        url = join_api(api_base_url, path)
        req = ExampleRequest(method="POST", url=url, json_body=payload)
        ex = generate_all(req)
        return CodePanel.update_all(ex["curl"], ex["python"], ex["java"], ex["javascript"])

    def _toggle(r2: bool):
        return (
            gr.update(visible=bool(r2)),
            gr.update(visible=not bool(r2)),
            gr.update(visible=bool(r2)),
        )

    def _call(api_base_url: str, timeout_seconds: float, r2: bool, p: str, s: float, key: str):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        payload: dict[str, Any] = {"prompt": p, "seed": int(s) if s is not None else 42}
        if r2 and key.strip():
            payload["key"] = key.strip()

        try:
            if r2:
                res = post_generate_image_to_r2(client=client, base_url=api_base_url, payload=payload)
                if res.status_code >= 400:
                    return f"HTTP {res.status_code}", None, safe_json_or_text(res.body_bytes)
                return "OK", None, safe_json_or_text(res.body_bytes)

            res = post_generate_image_png(client=client, base_url=api_base_url, payload=payload)
            if res.status_code >= 400:
                return f"HTTP {res.status_code}", None, safe_json_or_text(res.body_bytes)

            img_path = write_temp_bytes(res.body_bytes, suffix=".png")
            return "OK", img_path, None
        except Exception as exc:
            return "Error", None, {"error": str(exc)}

    # Events
    use_r2.change(_toggle, inputs=[use_r2], outputs=[r2_key, image_out, json_out])

    for comp in [base_url, use_r2, prompt, seed, r2_key]:
        comp.change(_example, inputs=[base_url, use_r2, prompt, seed, r2_key], outputs=panel.outputs())

    run.click(
        _call,
        inputs=[base_url, timeout, use_r2, prompt, seed, r2_key],
        outputs=[status, image_out, json_out],
    )
