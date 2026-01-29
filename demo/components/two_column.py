from __future__ import annotations

import gradio as gr


def two_column(*, left_title: str = "Demo", right_title: str = "Example code") -> tuple[gr.Column, gr.Column]:
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown(f"## {left_title}")
            left = gr.Column()
        with gr.Column(scale=5):
            gr.Markdown(f"## {right_title}")
            right = gr.Column()
    return left, right
