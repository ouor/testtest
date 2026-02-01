from __future__ import annotations

from typing import Any

import gradio as gr

from demo.code_examples.generators import ExampleRequest, generate_all
from demo.components.code_panel import CodePanel
from demo.components.r2_toggle import r2_toggle
from demo.components.result_viewers import safe_json_or_text
from demo.components.two_column import two_column
from demo.http_client import HttpClient
from demo.tabs.voice_cloning.api import post_generate_voice_mp3_multipart, post_generate_voice_to_r2
from demo.utils import join_api, write_temp_bytes


def build_voice_cloning_tab(*, base_url: gr.Textbox, timeout: gr.Number) -> None:
    left, right = two_column(left_title="Demo", right_title="Example code")

    with left:
        use_r2 = r2_toggle(default=False)

        # multipart mode
        ref_audio = gr.File(
            label="(multipart) ref_audio (mp3)", file_types=[".mp3"], visible=True, type="filepath"
        )

        ref_text = gr.Textbox(
            label="ref_text",
            value="아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든..",
            lines=2,
        )
        text = gr.Textbox(
            label="text",
            value="오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?",
            lines=2,
        )
        language = gr.Dropdown(label="language", choices=["Korean", "English", "Japanese", "Chinese"], value="Korean")

        # r2 mode
        user_id = gr.Textbox(label="(R2) user_id", placeholder="user123", visible=False)
        out_key = gr.Textbox(label="(R2) key (optional)", placeholder="voice/generated/custom.mp3", visible=False)

        run = gr.Button("Generate")
        status = gr.Markdown("")
        audio_out = gr.Audio(label="Generated audio", type="filepath", visible=True)
        json_out = gr.JSON(label="Response", visible=False)

    init_req = ExampleRequest(
        method="POST",
        url=join_api(base_url.value or "http://localhost:8000", "/v1/voice/generate"),
        multipart={
            "ref_audio": {"type": "file", "path": "path/to/ref.mp3"},
            "ref_text": "아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든..",
            "text": "오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?",
            "language": "Korean",
        },
    )
    init_examples = generate_all(init_req)

    with right:
        panel = CodePanel().build(initial=init_examples)

    def _toggle(r2: bool):
        return (
            gr.update(visible=not bool(r2)),  # ref_audio
            gr.update(visible=bool(r2)),  # user_id
            gr.update(visible=bool(r2)),  # out_key
            gr.update(visible=not bool(r2)),  # audio_out
            gr.update(visible=bool(r2)),  # json_out
            gr.update(visible=not bool(r2)),  # ref_text
        )

    def _example(
        api_base_url: str,
        r2: bool,
        ref_audio_path: str | None,
        rt: str,
        t: str,
        lang: str,
        uid: str,
        ok: str,
    ):
        if r2:
            payload: dict[str, Any] = {"user_id": uid or "user123", "text": t, "language": lang}
            if ok.strip():
                payload["key"] = ok.strip()
            url = join_api(api_base_url, "/v1/r2/voice/generate")
            req = ExampleRequest(method="POST", url=url, json_body=payload)
            ex = generate_all(req)
            return CodePanel.update_all(ex["curl"], ex["python"], ex["java"], ex["javascript"])

        url = join_api(api_base_url, "/v1/voice/generate")
        multipart = {
            "ref_audio": {"type": "file", "path": "path/to/ref.mp3"},
            "ref_text": rt,
            "text": t,
            "language": lang,
        }
        req = ExampleRequest(method="POST", url=url, multipart=multipart)
        ex = generate_all(req)
        return CodePanel.update_all(ex["curl"], ex["python"], ex["java"], ex["javascript"])

    def _call(
        api_base_url: str,
        timeout_seconds: float,
        r2: bool,
        ref_audio_path: str | None,
        rt: str,
        t: str,
        lang: str,
        uid: str,
        ok: str,
    ):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        try:
            if r2:
                if not uid.strip():
                    return "Error", None, {"error": "user_id is required in R2 mode"}
                payload: dict[str, Any] = {"user_id": uid.strip(), "text": t, "language": lang}
                if ok.strip():
                    payload["key"] = ok.strip()
                res = post_generate_voice_to_r2(client=client, base_url=api_base_url, payload=payload)
                if res.status_code >= 400:
                    return f"HTTP {res.status_code}", None, safe_json_or_text(res.body_bytes)
                return "OK", None, safe_json_or_text(res.body_bytes)

            if not ref_audio_path:
                return "Error", None, {"error": "ref_audio is required in multipart mode"}

            fields = {"ref_text": rt, "text": t, "language": lang}
            res = post_generate_voice_mp3_multipart(
                client=client, base_url=api_base_url, ref_audio_path=ref_audio_path, fields=fields
            )
            if res.status_code >= 400:
                return f"HTTP {res.status_code}", None, safe_json_or_text(res.body_bytes)

            mp3_path = write_temp_bytes(res.body_bytes, suffix=".mp3")
            return "OK", mp3_path, None
        except Exception as exc:
            return "Error", None, {"error": str(exc)}

    use_r2.change(_toggle, inputs=[use_r2], outputs=[ref_audio, user_id, out_key, audio_out, json_out, ref_text])

    for comp in [base_url, use_r2, ref_text, text, language, user_id, out_key]:
        comp.change(
            _example,
            inputs=[base_url, use_r2, ref_audio, ref_text, text, language, user_id, out_key],
            outputs=panel.outputs(),
        )

    run.click(
        _call,
        inputs=[base_url, timeout, use_r2, ref_audio, ref_text, text, language, user_id, out_key],
        outputs=[status, audio_out, json_out],
    )
