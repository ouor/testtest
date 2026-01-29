from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExampleRequest:
    method: str
    url: str
    json_body: dict[str, Any] | None = None
    multipart: dict[str, Any] | None = None
    note: str | None = None


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def curl_example(req: ExampleRequest) -> str:
    lines: list[str] = []
    if req.note:
        lines.append(f"# {req.note}")

    method = (req.method or "GET").upper()

    if req.json_body is not None:
        method_flag = "" if method == "POST" else f"-X {method} "
        lines.append(f"curl -s {method_flag}{req.url} \\")
        lines.append("  -H 'Content-Type: application/json' \\")
        lines.append(f"  -d '{_pretty_json(req.json_body)}'")
        return "\n".join(lines)

    if req.multipart is not None:
        method_flag = "" if method == "POST" else f"-X {method} "
        lines.append(f"curl -s {method_flag}{req.url} \\")
        for k, v in req.multipart.items():
            if isinstance(v, dict) and v.get("type") == "file":
                path = v.get("path") or "path/to/file"
                lines.append(f"  -F \"{k}=@{path}\" \\")
            else:
                lines.append(f"  -F \"{k}={v}\" \\")
        # remove trailing backslash for last line
        if lines and lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-2]
        return "\n".join(lines)

    if method == "GET":
        return "\n".join(lines + [f"curl -s {req.url}"])
    return "\n".join(lines + [f"curl -s -X {method} {req.url}"])


def python_example(req: ExampleRequest) -> str:
    method = (req.method or "GET").upper()
    if req.json_body is not None:
        if method not in {"POST", "PUT", "PATCH"}:
            method = "POST"
        return "\n".join(
            [
                "import requests",
                "",
                f"url = {req.url!r}",
                f"payload = {_pretty_json(req.json_body)}",
                f"resp = requests.{method.lower()}(url, json=payload)",
                "resp.raise_for_status()",
                "print(resp.json())",
            ]
        )

    if req.multipart is not None:
        # Assume POST multipart
        lines = [
            "import requests",
            "",
            f"url = {req.url!r}",
            "data = {}",
            "files = {}",
        ]
        for k, v in req.multipart.items():
            if isinstance(v, dict) and v.get("type") == "file":
                path = v.get("path") or "path/to/file"
                lines.append(f"files[{k!r}] = open({path!r}, 'rb')")
            else:
                lines.append(f"data[{k!r}] = {str(v)!r}")
        lines += [
            "resp = requests.post(url, data=data, files=files)",
            "resp.raise_for_status()",
            "print(resp.content)  # or resp.json()",
        ]
        return "\n".join(lines)

    # fallback GET
    call = "get" if method == "GET" else ("delete" if method == "DELETE" else "get")
    return "\n".join(
        [
            "import requests",
            "",
            f"resp = requests.{call}({req.url!r})",
            "resp.raise_for_status()",
            "print(resp.text)",
        ]
    )


def javascript_example(req: ExampleRequest) -> str:
    method = (req.method or "GET").upper()
    if req.json_body is not None:
        if method not in {"POST", "PUT", "PATCH"}:
            method = "POST"
        return "\n".join(
            [
                f"const url = {req.url!r};",
                f"const payload = {_pretty_json(req.json_body)};",
                "",
                "const resp = await fetch(url, {",
                f"  method: '{method}',",
                "  headers: { 'Content-Type': 'application/json' },",
                "  body: JSON.stringify(payload),",
                "});",
                "if (!resp.ok) throw new Error(await resp.text());",
                "console.log(await resp.json());",
            ]
        )

    if req.multipart is not None:
        lines = [
            f"const url = {req.url!r};",
            "const form = new FormData();",
        ]
        for k, v in req.multipart.items():
            if isinstance(v, dict) and v.get("type") == "file":
                path = v.get("path") or "path/to/file"
                lines.append(f"// NOTE: In browsers, use an <input type=\"file\"> to get a File object")
                lines.append(f"form.append({k!r}, /* File */ {path!r});")
            else:
                lines.append(f"form.append({k!r}, {str(v)!r});")
        lines += [
            "",
            "const resp = await fetch(url, { method: 'POST', body: form });",
            "if (!resp.ok) throw new Error(await resp.text());",
            "console.log(await resp.text());",
        ]
        return "\n".join(lines)

    return "\n".join(
        [
            f"const resp = await fetch({req.url!r}, {{ method: '{method}' }});",
            "if (!resp.ok) throw new Error(await resp.text());",
            "console.log(await resp.text());",
        ]
    )


def java_example(req: ExampleRequest) -> str:
    # Use Java 11+ HttpClient.
    method = (req.method or "GET").upper()
    if req.json_body is not None:
        body = _pretty_json(req.json_body)
        return "\n".join(
            [
                "import java.net.URI;",
                "import java.net.http.HttpClient;",
                "import java.net.http.HttpRequest;",
                "import java.net.http.HttpResponse;",
                "",
                "public class Main {",
                "  public static void main(String[] args) throws Exception {",
                f"    String url = {req.url!r};",
                f"    String json = {body!r};",
                "    HttpClient client = HttpClient.newHttpClient();",
                "    HttpRequest req = HttpRequest.newBuilder()",
                "      .uri(URI.create(url))",
                "      .header(\"Content-Type\", \"application/json\")",
                f"      .method(\"{method if method in {'POST','PUT','PATCH'} else 'POST'}\", HttpRequest.BodyPublishers.ofString(json))",
                "      .build();",
                "    HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());",
                "    if (resp.statusCode() >= 400) throw new RuntimeException(resp.body());",
                "    System.out.println(resp.body());",
                "  }",
                "}",
            ]
        )

    if req.multipart is not None:
        # Keep as a pragmatic snippet with placeholders.
        return "\n".join(
            [
                "// Java multipart example (pseudo-ish):",
                "// Consider using a library like OkHttp or Apache HttpClient for multipart uploads.",
                f"// URL: {req.url}",
                "// Fields:",
                *[f"// - {k}: {('FILE ' + (v.get('path') or 'path/to/file')) if isinstance(v, dict) and v.get('type')=='file' else v}" for k, v in req.multipart.items()],
            ]
        )

    method_stmt = "GET()" if method == "GET" else ("DELETE()" if method == "DELETE" else "GET()")
    return "\n".join(
        [
            "import java.net.URI;",
            "import java.net.http.HttpClient;",
            "import java.net.http.HttpRequest;",
            "import java.net.http.HttpResponse;",
            "",
            "public class Main {",
            "  public static void main(String[] args) throws Exception {",
            f"    String url = {req.url!r};",
            "    HttpClient client = HttpClient.newHttpClient();",
            "    HttpRequest req = HttpRequest.newBuilder()",
            "      .uri(URI.create(url))",
            f"      .{method_stmt}",
            "      .build();",
            "    HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());",
            "    System.out.println(resp.body());",
            "  }",
            "}",
        ]
    )


def generate_all(req: ExampleRequest) -> dict[str, str]:
    return {
        "curl": curl_example(req),
        "python": python_example(req),
        "java": java_example(req),
        "javascript": javascript_example(req),
    }
