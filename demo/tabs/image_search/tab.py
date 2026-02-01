from __future__ import annotations

from typing import Any

import gradio as gr

from demo.code_examples.generators import ExampleRequest, generate_all
from demo.components.code_panel import CodePanel
from demo.components.result_viewers import safe_json_or_text
from demo.components.two_column import two_column
from demo.http_client import HttpClient
from demo.tabs.image_search.api import delete_image, get_image_file_redirect, list_images, search_images, upload_image
from demo.utils import join_api, write_temp_bytes


def build_image_search_tab(*, base_url: gr.Textbox, timeout: gr.Number) -> None:
    left, right = two_column(left_title="Demo", right_title="Example code")

    with left:
        project_id = gr.Textbox(label="project_id", value="default")
        op = gr.Dropdown(
            label="Operation",
            choices=[
                "Upload Image",
                "List Images",
                "Search Images",
                "Get Image File (redirect)",
                "Delete Image",
            ],
            value="Upload Image",
        )

        upload_group = gr.Group(visible=True)
        with upload_group:
            upload_file = gr.File(label="Image file", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
            upload_btn = gr.Button("Upload")

        list_group = gr.Group(visible=False)
        with list_group:
            list_btn = gr.Button("List")

        search_group = gr.Group(visible=False)
        with search_group:
            query = gr.Textbox(label="query", value="스마트폰")
            limit = gr.Number(label="limit", value=5, precision=0)
            search_btn = gr.Button("Search")

        get_group = gr.Group(visible=False)
        with get_group:
            image_id_get = gr.Textbox(label="image_id")
            follow = gr.Checkbox(label="Follow redirect (download image)", value=True)
            get_btn = gr.Button("Get")
            redirect_url = gr.Textbox(label="redirect url (Location header)", interactive=False)
            downloaded_image = gr.Image(label="downloaded image", type="filepath")

        delete_group = gr.Group(visible=False)
        with delete_group:
            image_id_del = gr.Textbox(label="image_id")
            del_btn = gr.Button("Delete")

        out = gr.JSON(label="Response")
        status = gr.Markdown("")

    with right:
        init_examples = generate_all(
            ExampleRequest(
                method="POST",
                url=join_api(base_url.value or "http://localhost:8000", "/v1/projects/default/images"),
                multipart={"file": {"type": "file", "path": "path/to/image.jpg"}},
            )
        )
        panel = CodePanel().build(initial=init_examples)

    def _example(
        api_base_url: str,
        pid: str,
        operation: str,
        q: str,
        lim: float,
        image_id_get_val: str,
        image_id_del_val: str,
    ):
        image_id = image_id_get_val if operation == "Get Image File (redirect)" else image_id_del_val
        if operation == "Upload Image":
            req = ExampleRequest(
                method="POST",
                url=join_api(api_base_url, f"/v1/projects/{pid or 'default'}/images"),
                multipart={"file": {"type": "file", "path": "path/to/image.jpg"}},
            )
        elif operation == "List Images":
            req = ExampleRequest(method="GET", url=join_api(api_base_url, f"/v1/projects/{pid or 'default'}/images"))
        elif operation == "Search Images":
            req = ExampleRequest(
                method="POST",
                url=join_api(api_base_url, f"/v1/projects/{pid or 'default'}/images/search"),
                json_body={"query": q, "limit": int(lim) if lim is not None else 5},
            )
        elif operation == "Get Image File (redirect)":
            req = ExampleRequest(
                method="GET",
                url=join_api(api_base_url, f"/v1/projects/{pid or 'default'}/images/{image_id or '<id>'}/file"),
            )
        else:  # Delete
            req = ExampleRequest(
                method="DELETE",
                url=join_api(api_base_url, f"/v1/projects/{pid or 'default'}/images/{image_id or '<id>'}"),
            )

        ex = generate_all(req)
        return CodePanel.update_all(ex["curl"], ex["python"], ex["java"], ex["javascript"])

    def _op_change(operation: str):
        return (
            gr.update(visible=operation == "Upload Image"),
            gr.update(visible=operation == "List Images"),
            gr.update(visible=operation == "Search Images"),
            gr.update(visible=operation == "Get Image File (redirect)"),
            gr.update(visible=operation == "Delete Image"),
        )

    def _upload(api_base_url: str, timeout_seconds: float, pid: str, fpath: str | None):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        if not fpath:
            return {"error": "file is required"}, "Error"
        res = upload_image(client=client, base_url=api_base_url, project_id=pid or "default", image_path=fpath)
        return safe_json_or_text(res.body_bytes), f"HTTP {res.status_code}"

    def _list(api_base_url: str, timeout_seconds: float, pid: str):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        res = list_images(client=client, base_url=api_base_url, project_id=pid or "default")
        return safe_json_or_text(res.body_bytes), f"HTTP {res.status_code}"

    def _search(api_base_url: str, timeout_seconds: float, pid: str, q: str, lim: float):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        res = search_images(client=client, base_url=api_base_url, project_id=pid or "default", payload={"query": q, "limit": int(lim)})
        return safe_json_or_text(res.body_bytes), f"HTTP {res.status_code}"

    def _get(api_base_url: str, timeout_seconds: float, pid: str, image_id: str, follow_redirect: bool):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        res = get_image_file_redirect(
            client=client,
            base_url=api_base_url,
            project_id=pid or "default",
            image_id=image_id,
            follow=follow_redirect,
        )
        if not follow_redirect:
            return safe_json_or_text(res.body_bytes), res.headers.get("location") or "", None, f"HTTP {res.status_code}"

        if res.status_code >= 400:
            return safe_json_or_text(res.body_bytes), "", None, f"HTTP {res.status_code}"

        # When following redirects, the response body should be image bytes.
        ct = (res.headers.get("content-type") or "").lower()
        if "png" in ct:
            suffix = ".png"
        elif "jpeg" in ct or "jpg" in ct:
            suffix = ".jpg"
        elif "webp" in ct:
            suffix = ".webp"
        else:
            suffix = ".bin"
        img_path = write_temp_bytes(res.body_bytes, suffix=suffix)
        return {"downloaded": True, "content_type": res.headers.get("content-type")}, "", img_path, "OK"

    def _delete(api_base_url: str, timeout_seconds: float, pid: str, image_id: str):
        client = HttpClient(timeout_seconds=float(timeout_seconds))
        res = delete_image(client=client, base_url=api_base_url, project_id=pid or "default", image_id=image_id)
        if res.status_code == 204:
            return {"deleted": True}, "OK"
        return safe_json_or_text(res.body_bytes), f"HTTP {res.status_code}"

    op.change(_op_change, inputs=[op], outputs=[upload_group, list_group, search_group, get_group, delete_group])

    for comp in [base_url, project_id, op, query, limit, image_id_get, image_id_del]:
        comp.change(
            _example,
            inputs=[base_url, project_id, op, query, limit, image_id_get, image_id_del],
            outputs=panel.outputs(),
        )

    upload_btn.click(_upload, inputs=[base_url, timeout, project_id, upload_file], outputs=[out, status])
    list_btn.click(_list, inputs=[base_url, timeout, project_id], outputs=[out, status])
    search_btn.click(_search, inputs=[base_url, timeout, project_id, query, limit], outputs=[out, status])
    get_btn.click(
        _get,
        inputs=[base_url, timeout, project_id, image_id_get, follow],
        outputs=[out, redirect_url, downloaded_image, status],
    )
    del_btn.click(_delete, inputs=[base_url, timeout, project_id, image_id_del], outputs=[out, status])

    
