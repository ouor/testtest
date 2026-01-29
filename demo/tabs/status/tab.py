from __future__ import annotations

import gradio as gr

from demo.code_examples.generators import ExampleRequest, generate_all
from demo.components.code_panel import CodePanel
from demo.components.result_viewers import safe_json_or_text
from demo.components.two_column import two_column
from demo.http_client import HttpClient
from demo.utils import join_api


def build_status_tab(*, base_url: gr.Textbox, timeout: gr.Number) -> None:
    left, right = two_column(left_title="Demo", right_title="Example code")

    with left:
        endpoint = gr.Dropdown(
            label="Endpoint",
            choices=["/healthz", "/readyz"],
            value="/healthz",
        )
        run = gr.Button("Call")
        out = gr.JSON(label="Response")

    init_url = join_api(base_url.value or "http://localhost:8000", "/healthz")
    init_examples = generate_all(ExampleRequest(method="GET", url=init_url, note="Status check"))

    with right:
        panel = CodePanel().build(initial=init_examples)

    def _make_examples(api_base_url: str, ep: str):
        url = join_api(api_base_url, ep)
        req = ExampleRequest(method="GET", url=url, note="Status check")
        ex = generate_all(req)
        return CodePanel.update_all(ex["curl"], ex["python"], ex["java"], ex["javascript"])

    def _call(api_base_url: str, timeout_seconds: float, ep: str):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        url = join_api(api_base_url, ep)
        res = client.get(url)
        return safe_json_or_text(res.body_bytes)

    # Events
    endpoint.change(_make_examples, inputs=[base_url, endpoint], outputs=panel.outputs())
    base_url.change(_make_examples, inputs=[base_url, endpoint], outputs=panel.outputs())
    run.click(_call, inputs=[base_url, timeout, endpoint], outputs=[out])
