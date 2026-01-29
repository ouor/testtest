from __future__ import annotations

import gradio as gr


def r2_toggle(*, default: bool = False) -> gr.Checkbox:
    return gr.Checkbox(label="R2 요청", value=bool(default))
