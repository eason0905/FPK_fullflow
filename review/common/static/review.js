(function () {
  const config = window.REVIEW_CONFIG || {};
  const notesPath = config.notesPath || "";
  const historyPath = config.historyPath || "";
  const saveTimers = new Map();
  let notes = {};

  function byCaseId(caseId) {
    return document.querySelector(`[data-case-id="${CSS.escape(caseId)}"]`);
  }

  function apiUrl() {
    return `/api/review-notes?notes_path=${encodeURIComponent(notesPath)}`;
  }

  async function loadNotes() {
    if (!notesPath) return;
    const response = await fetch(apiUrl(), { cache: "no-store" });
    if (!response.ok) throw new Error(`Cannot load notes: ${response.status}`);
    const payload = await response.json();
    notes = payload.items || {};
    restoreNotes();
    updateCounts();
  }

  function restoreNotes() {
    for (const [caseId, note] of Object.entries(notes)) {
      const card = byCaseId(caseId);
      if (!card) continue;
      const textarea = card.querySelector('[data-field="issue_text"]');
      const category = card.querySelector('[data-field="category"]');
      const status = card.querySelector('[data-field="status"]');
      if (textarea) textarea.value = note.issue_text || "";
      if (category) category.value = note.category || "";
      if (status) status.value = note.status || "";
      setSaveState(card, note.updated_at ? `saved ${formatTime(note.updated_at)}` : "saved");
    }
  }

  function collectPayload(card) {
    const caseId = card.dataset.caseId;
    const textarea = card.querySelector('[data-field="issue_text"]');
    const category = card.querySelector('[data-field="category"]');
    const status = card.querySelector('[data-field="status"]');
    return {
      case_id: caseId,
      issue_text: textarea ? textarea.value : "",
      category: category ? category.value : "",
      status: status ? status.value : "",
      page: config.pageIndex || null,
      title: card.dataset.title || "",
      part_number: card.dataset.partNumber || "",
      file_name: card.dataset.fileName || "",
      view: card.dataset.view || "",
    };
  }

  async function saveCard(card) {
    const payload = collectPayload(card);
    setSaveState(card, "saving...");
    const response = await fetch("/api/review-notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        notes_path: notesPath,
        history_path: historyPath,
        gallery_id: config.galleryId || "review",
        run_id: config.runId || "",
        payload,
      }),
    });
    if (!response.ok) {
      setSaveState(card, `save failed (${response.status})`);
      return;
    }
    const data = await response.json();
    notes = data.items || {};
    setSaveState(card, `saved ${formatTime(data.updated_at)}`);
    updateCounts();
  }

  function scheduleSave(card) {
    const caseId = card.dataset.caseId;
    setSaveState(card, "editing...");
    clearTimeout(saveTimers.get(caseId));
    saveTimers.set(
      caseId,
      setTimeout(() => {
        saveCard(card).catch((error) => {
          setSaveState(card, error.message || "save failed");
        });
      }, 450),
    );
  }

  function setSaveState(card, value) {
    const node = card.querySelector(".save-state");
    if (node) node.textContent = value;
  }

  function formatTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function updateCounts() {
    const count = Object.keys(notes).filter((caseId) => {
      const note = notes[caseId] || {};
      return note.issue_text || note.category || note.status;
    }).length;
    document.querySelectorAll("[data-reviewed-count]").forEach((node) => {
      node.textContent = String(count);
    });
  }

  function applyFilters() {
    const text = (document.querySelector("#filter-text")?.value || "").trim().toLowerCase();
    const status = document.querySelector("#filter-status")?.value || "";
    const category = document.querySelector("#filter-category")?.value || "";
    for (const card of document.querySelectorAll(".review-card")) {
      const caseId = card.dataset.caseId;
      const note = notes[caseId] || {};
      const haystack = [
        card.dataset.title,
        card.dataset.partNumber,
        card.dataset.fileName,
        card.dataset.view,
        note.issue_text,
        note.category,
        note.status,
      ]
        .join(" ")
        .toLowerCase();
      const textOk = !text || haystack.includes(text);
      const statusOk = !status || note.status === status;
      const categoryOk = !category || note.category === category;
      card.classList.toggle("hidden", !(textOk && statusOk && categoryOk));
    }
  }

  function downloadNotes() {
    const blob = new Blob([JSON.stringify({ items: notes }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "notes.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function bind() {
    document.querySelectorAll(".review-card").forEach((card) => {
      card.querySelectorAll("textarea, select").forEach((node) => {
        node.addEventListener("input", () => scheduleSave(card));
        node.addEventListener("change", () => scheduleSave(card));
      });
      card.querySelector('[data-action="save-now"]')?.addEventListener("click", () => {
        saveCard(card).catch((error) => setSaveState(card, error.message || "save failed"));
      });
    });

    document.querySelectorAll("[data-filter]").forEach((node) => {
      node.addEventListener("input", applyFilters);
      node.addEventListener("change", applyFilters);
    });
    document.querySelector("[data-action='download-notes']")?.addEventListener("click", downloadNotes);
  }

  bind();
  loadNotes().catch((error) => {
    const banner = document.querySelector("#review-server-state");
    if (banner) banner.textContent = `notes server unavailable: ${error.message}`;
  });
  window.applyReviewFilters = applyFilters;
})();

