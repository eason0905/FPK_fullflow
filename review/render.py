from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .schema import ReviewCase


def render_root_index(
    index_path: Path,
    cases_by_task: dict[str, list[ReviewCase]],
    *,
    title: str = "LLM Error Review",
    description: str = "每個 task 都有獨立 reviewer；標記會存在瀏覽器 localStorage，可從 task 頁匯出 JSONL。",
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(len(cases) for cases in cases_by_task.values())
    rows = []
    for task, cases in cases_by_task.items():
        rows.append(
            f"<tr><td><a href=\"{html.escape(task)}/index.html\">{html.escape(task)}</a></td>"
            f"<td>{len(cases)}</td><td>{reason_summary(cases)}</td></tr>"
        )
    rows.append(f"<tr><td><a href=\"all/index.html\">all</a></td><td>{total}</td><td>all tasks</td></tr>")
    body = f"""
<h1>{html.escape(title)}</h1>
<p class="muted">{html.escape(description)}</p>
<table>
  <thead><tr><th>Task</th><th>Error cases</th><th>Reasons</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""
    index_path.write_text(page(title, body), encoding="utf-8")


def render_task_gallery(task: str, cases: list[ReviewCase], index_path: Path, *, gallery_root: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    case_json = json.dumps([case.to_dict() for case in cases], ensure_ascii=False)
    cards = "\n".join(render_case(case, index_path.parent, gallery_root) for case in cases)
    body = f"""
<div class="toolbar">
  <div>
    <h1>{html.escape(task)} reviewer</h1>
    <p class="muted">{len(cases)} error cases. 判斷是 model error、annotation error、ambiguous、parser issue 或 ignore。</p>
  </div>
  <div class="actions">
    <a class="button" href="../index.html">Index</a>
    <button onclick="exportNotes()">Export notes JSONL</button>
    <button onclick="clearNotes()">Clear local notes</button>
  </div>
</div>
<div class="summary">
  <span>Total: {len(cases)}</span>
  <span>{reason_summary(cases)}</span>
  <span id="note-count">Reviewed: 0</span>
</div>
<div class="filters">
  <label>Reason <select id="reason-filter" onchange="applyFilters()"><option value="">all</option>{reason_options(cases)}</select></label>
  <label>Decision <select id="decision-filter" onchange="applyFilters()"><option value="">all</option><option>model_error</option><option>annotation_error</option><option>ambiguous</option><option>parser_issue</option><option>ignore</option></select></label>
</div>
<script id="cases-data" type="application/json">{html.escape(case_json)}</script>
<section class="cases">
{cards or '<p class="muted">No error cases.</p>'}
</section>
{script()}
"""
    index_path.write_text(page(f"{task} reviewer", body), encoding="utf-8")


def render_case(case: ReviewCase, page_dir: Path, gallery_root: Path) -> str:
    payload = case.to_dict()
    image_src = image_source(case, page_dir, gallery_root)
    expected = html.escape(case.expected)
    predicted = html.escape(case.predicted)
    prompt = html.escape(trim_prompt(case.prompt))
    metadata = html.escape(json.dumps(case.metadata, ensure_ascii=False, indent=2))
    source = html.escape(case.source_path)
    tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in case.tags)
    context = render_case_context(case)
    image_block = (
        f"<a href=\"{html.escape(image_src)}\" target=\"_blank\"><img src=\"{html.escape(image_src)}\" loading=\"lazy\"></a>"
        if image_src
        else "<div class=\"missing-image\">missing image</div>"
    )
    decisions = "".join(
        f"<label><input type=\"radio\" name=\"decision-{html.escape(case.case_id)}\" "
        f"value=\"{value}\" onchange=\"saveDecision('{js_escape(case.case_id)}')\"> {value}</label>"
        for value in ("model_error", "annotation_error", "ambiguous", "parser_issue", "ignore")
    )
    return f"""
<article class="case" data-case-id="{html.escape(case.case_id)}" data-reason="{html.escape(case.reason)}">
  <div class="image-pane">{image_block}</div>
  <div class="detail-pane">
    <div class="case-head">
      <h2>{html.escape(case.case_id)}</h2>
      <span class="reason">{html.escape(case.reason)}</span>
    </div>
    <div class="tags">{tags}</div>
    {context}
    <div class="compare">
      <div><h3>Expected</h3><pre>{expected}</pre></div>
      <div><h3>Predicted</h3><pre>{predicted}</pre></div>
    </div>
    <details><summary>Prompt</summary><pre>{prompt}</pre></details>
    <details><summary>Metadata</summary><pre>{metadata}</pre></details>
    <p class="path">{source}</p>
    <div class="review-box">
      <div class="decision">{decisions}</div>
      <textarea placeholder="note..." oninput="saveDecision('{js_escape(case.case_id)}')"></textarea>
    </div>
  </div>
</article>
"""


def render_case_context(case: ReviewCase) -> str:
    part_number = case.metadata.get("part_number") or ""
    annotation_path = case.metadata.get("annotation_path") or ""
    file_name = case.metadata.get("annotation_file_name") or case.metadata.get("source_file_name") or ""
    source_image_stem = case.metadata.get("source_image_stem") or ""
    if not part_number and not annotation_path and not source_image_stem:
        return ""

    cells = []
    if part_number:
        cells.append(f"<div><h3>Part</h3><pre>{html.escape(str(part_number))}</pre></div>")
    if file_name:
        cells.append(f"<div><h3>File name</h3><pre>{html.escape(str(file_name))}</pre></div>")
    if source_image_stem:
        cells.append(f"<div><h3>Source Image</h3><pre>{html.escape(str(source_image_stem))}</pre></div>")
    if annotation_path:
        cells.append(f"<div><h3>Original JSON</h3><pre>{html.escape(str(annotation_path))}</pre></div>")
    return f"<div class=\"context-grid\">{''.join(cells)}</div>"


def image_source(case: ReviewCase, page_dir: Path, gallery_root: Path) -> str:
    if case.image_rel_path:
        image_path = gallery_root / case.image_rel_path
        return Path("../" * len(page_dir.relative_to(gallery_root).parts)).joinpath(
            image_path.relative_to(gallery_root)
        ).as_posix()
    if case.image_path:
        return quote(case.image_path)
    return ""


def reason_summary(cases: list[ReviewCase]) -> str:
    counts = Counter(case.reason for case in cases)
    return ", ".join(f"{html.escape(reason)}: {count}" for reason, count in counts.most_common()) or "-"


def reason_options(cases: list[ReviewCase]) -> str:
    reasons = sorted({case.reason for case in cases})
    return "".join(f"<option>{html.escape(reason)}</option>" for reason in reasons)


def trim_prompt(prompt: str, limit: int = 5000) -> str:
    if len(prompt) <= limit:
        return prompt
    return prompt[:limit] + "\n...\n[trimmed]"


def js_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #15171a; color: #e8edf2; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
    a {{ color: #62a8ff; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; }}
    h3 {{ color: #aeb7c2; font-size: 13px; margin-bottom: 6px; }}
    table {{ border-collapse: collapse; width: min(980px, calc(100vw - 32px)); margin: 20px 16px; }}
    th, td {{ border-bottom: 1px solid #333942; padding: 10px; text-align: left; }}
    .muted {{ color: #aeb7c2; }}
    .toolbar {{ align-items: flex-start; border-bottom: 1px solid #2b3038; display: flex; gap: 16px; justify-content: space-between; padding: 16px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button, .button {{ background: #26313d; border: 1px solid #3c4a59; border-radius: 6px; color: #e8edf2; cursor: pointer; display: inline-block; font-size: 13px; padding: 8px 10px; text-decoration: none; }}
    .summary, .filters {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 16px; }}
    .summary span {{ background: #20252c; border: 1px solid #303741; border-radius: 6px; padding: 6px 8px; }}
    select {{ background: #20252c; border: 1px solid #3c4a59; border-radius: 6px; color: #e8edf2; padding: 6px; }}
    .cases {{ display: grid; gap: 12px; padding: 0 16px 24px; }}
    .case {{ background: #1c2026; border: 1px solid #303741; border-radius: 8px; display: grid; grid-template-columns: minmax(280px, 45%) minmax(360px, 1fr); overflow: hidden; }}
    .image-pane {{ align-items: center; background: #f8fafc; display: flex; justify-content: center; min-height: 300px; }}
    .image-pane img {{ display: block; max-height: 640px; max-width: 100%; object-fit: contain; }}
    .missing-image {{ color: #64748b; }}
    .detail-pane {{ display: grid; gap: 12px; padding: 14px; }}
    .case-head {{ align-items: center; display: flex; gap: 12px; justify-content: space-between; }}
    .reason {{ background: #5b2631; border: 1px solid #a74455; border-radius: 999px; color: #ffd7df; padding: 4px 8px; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tags span {{ background: #253041; border: 1px solid #35465f; border-radius: 999px; color: #b9d5ff; font-size: 12px; padding: 3px 8px; }}
    .context-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .compare {{ display: grid; gap: 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    pre {{ background: #101216; border: 1px solid #303741; border-radius: 6px; color: #f4f7fa; margin: 0; overflow: auto; padding: 10px; white-space: pre-wrap; }}
    details summary {{ color: #c9d2dc; cursor: pointer; margin-bottom: 6px; }}
    .path {{ color: #8c98a7; font-size: 12px; overflow-wrap: anywhere; }}
    .review-box {{ border-top: 1px solid #303741; display: grid; gap: 8px; padding-top: 10px; }}
    .decision {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    textarea {{ background: #101216; border: 1px solid #303741; border-radius: 6px; color: #e8edf2; min-height: 70px; padding: 8px; resize: vertical; }}
    @media (max-width: 900px) {{ .case {{ grid-template-columns: 1fr; }} .compare, .context-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def script() -> str:
    return """
<script>
const cases = JSON.parse(document.getElementById('cases-data').textContent);
const storageKey = 'fpk_llm_review:' + location.pathname;

function loadNotes() {
  try { return JSON.parse(localStorage.getItem(storageKey) || '{}'); }
  catch { return {}; }
}

function saveNotes(notes) {
  localStorage.setItem(storageKey, JSON.stringify(notes));
  updateReviewedCount();
}

function saveDecision(caseId) {
  const card = document.querySelector(`[data-case-id="${CSS.escape(caseId)}"]`);
  const selected = card.querySelector('input[type="radio"]:checked');
  const note = card.querySelector('textarea').value;
  const notes = loadNotes();
  notes[caseId] = {
    case_id: caseId,
    decision: selected ? selected.value : '',
    note,
    updated_at: new Date().toISOString()
  };
  if (!notes[caseId].decision && !notes[caseId].note) delete notes[caseId];
  saveNotes(notes);
}

function restoreNotes() {
  const notes = loadNotes();
  for (const [caseId, payload] of Object.entries(notes)) {
    const card = document.querySelector(`[data-case-id="${CSS.escape(caseId)}"]`);
    if (!card) continue;
    if (payload.decision) {
      const radio = card.querySelector(`input[value="${CSS.escape(payload.decision)}"]`);
      if (radio) radio.checked = true;
    }
    card.querySelector('textarea').value = payload.note || '';
  }
  updateReviewedCount();
}

function updateReviewedCount() {
  document.getElementById('note-count').textContent = `Reviewed: ${Object.keys(loadNotes()).length}`;
}

function exportNotes() {
  const notes = Object.values(loadNotes());
  const blob = new Blob([notes.map(row => JSON.stringify(row)).join('\\n') + (notes.length ? '\\n' : '')], {type: 'application/jsonl'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'review_notes.jsonl';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function clearNotes() {
  if (!confirm('Clear local review notes for this page?')) return;
  localStorage.removeItem(storageKey);
  document.querySelectorAll('input[type="radio"]').forEach(input => input.checked = false);
  document.querySelectorAll('textarea').forEach(input => input.value = '');
  updateReviewedCount();
  applyFilters();
}

function applyFilters() {
  const reason = document.getElementById('reason-filter').value;
  const decision = document.getElementById('decision-filter').value;
  const notes = loadNotes();
  document.querySelectorAll('.case').forEach(card => {
    const reasonOk = !reason || card.dataset.reason === reason;
    const note = notes[card.dataset.caseId] || {};
    const decisionOk = !decision || note.decision === decision;
    card.style.display = reasonOk && decisionOk ? '' : 'none';
  });
}

restoreNotes();
</script>
"""
