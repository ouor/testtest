from __future__ import annotations

import gradio as gr


class CodePanel:
    def __init__(self) -> None:
        self.curl: gr.Code | None = None
        self.python: gr.Code | None = None
        self.java: gr.Code | None = None
        self.javascript: gr.Code | None = None

    def build(self, *, initial: dict[str, str] | None = None) -> "CodePanel":
        initial = initial or {}
        with gr.Tabs():
            with gr.Tab(label="curl"):
                self.curl = gr.Code(label="curl", language="shell", interactive=False, value=initial.get("curl", ""))
            with gr.Tab(label="python"):
                self.python = gr.Code(
                    label="python", language="python", interactive=False, value=initial.get("python", "")
                )
            with gr.Tab(label="java"):
                self.java = gr.Code(label="java", language="cpp", interactive=False, value=initial.get("java", ""))
            with gr.Tab(label="javascript"):
                self.javascript = gr.Code(
                    label="javascript",
                    language="javascript",
                    interactive=False,
                    value=initial.get("javascript", ""),
                )
        return self

    def outputs(self) -> list[gr.Code]:
        assert self.curl and self.python and self.java and self.javascript
        return [self.curl, self.python, self.java, self.javascript]

    @staticmethod
    def update_all(curl: str, python: str, java: str, javascript: str):
        return (
            gr.update(value=curl),
            gr.update(value=python),
            gr.update(value=java),
            gr.update(value=javascript),
        )
