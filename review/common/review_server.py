from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve fullflow review galleries and persist notes.")
    parser.add_argument("--root", type=Path, required=True, help="FPK_PJ_fullflow root to serve.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def run_server(root: Path, *, host: str = "127.0.0.1", port: int = 8767) -> None:
    root = root.resolve()
    handler = build_handler(root)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {root} at http://{host}:{port}/", flush=True)
    server.serve_forever()


def build_handler(root: Path) -> type[SimpleHTTPRequestHandler]:
    class ReviewRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/review-notes":
                query = parse_qs(parsed.query)
                notes_path = query.get("notes_path", [""])[0]
                self.send_json(load_notes(root, notes_path))
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/review-notes":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = save_note(root, payload)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self.send_json(result)

        def send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ReviewRequestHandler


def resolve_review_file(root: Path, value: str, *, expected_name: str) -> Path:
    if not value:
        raise ValueError("missing review file path")
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError(f"path escapes review root: {value}")
    if path.name != expected_name:
        raise ValueError(f"expected {expected_name}, got {path.name}")
    return path


def load_notes(root: Path, notes_path: str) -> dict[str, Any]:
    path = resolve_review_file(root, notes_path, expected_name="notes.json")
    if not path.exists():
        return {"gallery_id": "", "run_id": "", "updated_at": None, "items": {}}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_note(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    notes_path = resolve_review_file(root, str(request.get("notes_path") or ""), expected_name="notes.json")
    history_value = str(request.get("history_path") or "")
    history_path = (
        resolve_review_file(root, history_value, expected_name="notes_history.jsonl") if history_value else None
    )
    payload = dict(request.get("payload") or {})
    case_id = str(payload.get("case_id") or "")
    if not case_id:
        raise ValueError("missing case_id")

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes = load_existing_notes(notes_path)
    notes["gallery_id"] = request.get("gallery_id") or notes.get("gallery_id") or ""
    notes["run_id"] = request.get("run_id") or notes.get("run_id") or ""
    updated_at = utc_now()
    notes["updated_at"] = updated_at
    items = notes.setdefault("items", {})

    note = {
        "case_id": case_id,
        "issue_text": str(payload.get("issue_text") or ""),
        "category": str(payload.get("category") or ""),
        "status": str(payload.get("status") or ""),
        "page": payload.get("page"),
        "title": str(payload.get("title") or ""),
        "part_number": str(payload.get("part_number") or ""),
        "file_name": str(payload.get("file_name") or ""),
        "view": str(payload.get("view") or ""),
        "updated_at": updated_at,
    }
    if note["issue_text"] or note["category"] or note["status"]:
        items[case_id] = note
    else:
        items.pop(case_id, None)

    atomic_write_json(notes_path, notes)
    if history_path is not None:
        append_history(history_path, notes["gallery_id"], notes["run_id"], note)
    return notes


def load_existing_notes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"gallery_id": "", "run_id": "", "updated_at": None, "items": {}}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temp_path = Path(file.name)
    temp_path.replace(path)


def append_history(path: Path, gallery_id: str, run_id: str, note: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "gallery_id": gallery_id,
        "run_id": run_id,
        "updated_at": note.get("updated_at"),
        "case_id": note.get("case_id"),
        "note": note,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False))
        file.write("\n")


def main() -> None:
    args = parse_args()
    run_server(args.root, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

