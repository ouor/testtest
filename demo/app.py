from __future__ import annotations

import gradio as gr

from demo.config import DemoConfig
from demo.tabs.image_generation.tab import build_image_generation_tab
from demo.tabs.image_search.tab import build_image_search_tab
from demo.tabs.status.tab import build_status_tab
from demo.tabs.voice_cloning.tab import build_voice_cloning_tab


def build_app() -> gr.Blocks:
    cfg = DemoConfig.from_env()

    with gr.Blocks(title="Common AI Server Demo & API") as demo:
        gr.Markdown("# Common AI Server Demo & API\n")

        with gr.Row():
            base_url = gr.Textbox(
                label="API Base URL",
                value=cfg.api_base_url,
                interactive=True,
                placeholder="http://localhost:8000",
            )
            timeout = gr.Number(label="Timeout (sec)", value=float(cfg.timeout_seconds), precision=0)

        gr.Markdown(
            "왼쪽: Demo / 오른쪽: 호출 예제(curl, python, java, javascript)"
        )

        with gr.Tabs():
            with gr.Tab(label="Status"):
                build_status_tab(base_url=base_url, timeout=timeout)

            with gr.Tab(label="Image Generation"):
                build_image_generation_tab(base_url=base_url, timeout=timeout)

            with gr.Tab(label="Voice Cloning"):
                build_voice_cloning_tab(base_url=base_url, timeout=timeout)

            with gr.Tab(label="Image Search"):
                build_image_search_tab(base_url=base_url, timeout=timeout)

    return demo


def main() -> None:
    demo = build_app()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
