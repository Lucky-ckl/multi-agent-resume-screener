"use strict";

const form = document.getElementById("screen-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const submitBtn = document.getElementById("submit-btn");

const SECTION_ORDER = ["skills", "experience", "projects", "education"];

/* 板块与匹配结论的中文展示名（键与后端字段保持一致）。 */
const SECTION_LABELS = {
  skills: "技能",
  experience: "经历",
  projects: "项目",
  education: "教育",
};

const VERDICT_LABELS = {
  strong_fit: "强匹配",
  moderate_fit: "中等匹配",
  weak_fit: "弱匹配",
};

const MODE_LABELS = { candidate: "候选人", recruiter: "招聘者" };

/* 展示给候选人的常青"招聘官速赢"技巧（这些事现在就能做，
   且纯文本检查器无法从提取出的 PDF 中验证）。 */
const EVERGREEN_QUICKWINS = [
  "加粗招聘官会扫读的工具、数据和关键词。",
  "每条经历用强动作动词开头，并带上具体数字。",
  "每条经历控制在一行以内，删掉废话。",
  "在适用处原样使用职位描述里的技能术语。",
];

/* 一个安全设置文本的小型 DOM 辅助函数（对不可信数据不使用 innerHTML）。 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function showStatus(message, isError) {
  statusEl.hidden = false;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function clearStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
  statusEl.classList.remove("error");
}

function pct(value) {
  return Math.round((Number(value) || 0) * 100);
}

function renderBreakdown(breakdown) {
  const wrap = el("div", "breakdown");
  SECTION_ORDER.forEach((section) => {
    const value = breakdown && section in breakdown ? breakdown[section] : 0;
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-label", SECTION_LABELS[section] || section));

    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = pct(value) + "%";
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("span", "bar-value", pct(value) + "%"));
    wrap.appendChild(row);
  });
  return wrap;
}

function renderList(title, items, emptyText) {
  const frag = document.createDocumentFragment();
  frag.appendChild(el("p", "section-title", title));
  if (items && items.length) {
    const ul = el("ul");
    items.forEach((item) => ul.appendChild(el("li", null, item)));
    frag.appendChild(ul);
  } else {
    frag.appendChild(el("p", "empty", emptyText));
  }
  return frag;
}

function renderReasons(candidate) {
  const reasons = candidate.section_reasons || {};
  const evidence = candidate.section_evidence || {};
  const sections = SECTION_ORDER.filter((s) => reasons[s]);
  if (!sections.length) return null;

  const wrap = el("div", "reasons");
  wrap.appendChild(el("p", "section-title", "排名理由"));
  sections.forEach((section) => {
    const row = el("div", "reason-row");
    row.appendChild(el("span", "reason-section", SECTION_LABELS[section] || section));
    const body = el("div", "reason-body");
    body.appendChild(el("span", "reason-text", reasons[section]));
    const ev = evidence[section];
    if (ev && ev.length) {
      body.appendChild(el("span", "reason-evidence", "证据：" + ev.join("；")));
    }
    row.appendChild(body);
    wrap.appendChild(row);
  });
  return wrap;
}

function renderCandidate(candidate, index, mode) {
  const card = el("div", "candidate");

  const head = el("div", "candidate-head");
  const titleWrap = el("div");
  const name = el("h3", "candidate-name", candidate.candidate_name || "候选人");
  const file = el("span", "candidate-file", "  " + (candidate.filename || ""));
  name.appendChild(file);
  titleWrap.appendChild(name);
  head.appendChild(titleWrap);

  const scoreWrap = el("div");
  scoreWrap.appendChild(el("span", "score", pct(candidate.score) + "%  "));
  const verdict = candidate.verdict || "weak_fit";
  scoreWrap.appendChild(el("span", "badge " + verdict, VERDICT_LABELS[verdict] || verdict));
  head.appendChild(scoreWrap);

  card.appendChild(head);
  card.appendChild(renderBreakdown(candidate.breakdown));

  const reasons = renderReasons(candidate);
  if (reasons) card.appendChild(reasons);

  // 第 1 桶 — 长期积累：为该岗位需要发展的技能/经验。
  const build = el("div", "bucket build-over-time");
  build.appendChild(el("p", "bucket-title", "长期积累 — 面向该岗位"));
  build.appendChild(
    renderList("与岗位的差距", candidate.gaps, "未发现明显差距。")
  );
  build.appendChild(
    renderList(
      "需要积累的技能与经验",
      candidate.suggestions,
      "暂无建议。"
    )
  );
  card.appendChild(build);

  // 第 2 桶 — 立刻修复：即时、确定性的简历修改，展示在下方。
  const fix = el("div", "bucket fix-now");
  fix.appendChild(el("p", "bucket-title", "立即修改 — 快速简历调整"));
  if (candidate.hygiene_score !== null && candidate.hygiene_score !== undefined) {
    fix.appendChild(
      el("p", "hygiene-line", "简历健康度：" + pct(candidate.hygiene_score) + "%")
    );
  }
  const issues = candidate.hygiene_issues || [];
  if (issues.length) {
    const ul = el("ul");
    issues.forEach((i) => ul.appendChild(el("li", null, i)));
    fix.appendChild(ul);
  } else {
    fix.appendChild(el("p", "empty", "未发现需立即修改的问题——简历很干净。"));
  }
  if (mode === "candidate") {
    fix.appendChild(el("p", "tips-title", "招聘官速赢技巧"));
    const tips = el("ul", "tips");
    EVERGREEN_QUICKWINS.forEach((t) => tips.appendChild(el("li", null, t)));
    fix.appendChild(tips);
  }
  card.appendChild(fix);

  return card;
}

function renderResults(data) {
  resultsEl.innerHTML = "";
  const candidates = data.candidates || [];

  const summary = el(
    "p",
    "results-summary",
    `共 ${candidates.length} 份结果 · 模式：${MODE_LABELS[data.mode] || data.mode}` +
      (data.job_title ? ` · 岗位：${data.job_title}` : "")
  );
  resultsEl.appendChild(summary);

  candidates.forEach((c, i) => resultsEl.appendChild(renderCandidate(c, i, data.mode)));
}

async function handleSubmit(event) {
  event.preventDefault();
  resultsEl.innerHTML = "";

  const files = document.getElementById("resumes").files;
  if (!files || files.length === 0) {
    showStatus("请至少选择一份简历 PDF。", true);
    return;
  }

  const formData = new FormData(form);
  submitBtn.disabled = true;
  showStatus("筛选中……若服务刚唤醒，首次请求约需 30 秒。");

  try {
    const resp = await fetch("/screen", { method: "POST", body: formData });
    if (!resp.ok) {
      let detail = "请求失败（HTTP " + resp.status + "）。";
      try {
        const err = await resp.json();
        if (err.detail) detail = typeof err.detail === "string"
          ? err.detail
          : JSON.stringify(err.detail);
      } catch (_) {
        /* 忽略 JSON 解析错误 */
      }
      showStatus(detail, true);
      return;
    }
    const data = await resp.json();
    clearStatus();
    renderResults(data);
  } catch (e) {
    showStatus("网络错误：" + e.message, true);
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", handleSubmit);
