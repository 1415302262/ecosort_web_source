"use strict";
const $ = (id) => document.getElementById(id);
const names = {
  brick_ceramic: "砖块 / 陶瓷",
  e_waste: "电子废弃物",
  foam: "泡沫塑料",
  glass: "玻璃",
  haz_battery: "废电池",
  haz_device: "灯泡 / 温度计",
  haz_medicine: "废弃药品",
  hygiene_contaminated: "受污染卫生用品",
  metal: "金属",
  organic_food: "厨余 / 有机物",
  paper_family: "纸类",
  plastic: "塑料",
  small_hard_items: "小型硬质杂物",
  textile: "纺织品",
};
const coarse = {
  recyclable: "可回收物",
  hazardous: "有害垃圾",
  kitchen: "厨余垃圾",
  other: "其他垃圾",
};
const statusNames = {
  pending: "待审核",
  approved: "已采纳",
  rejected: "已驳回",
};
let sessionInfo = null,
  historyPage = 1,
  reviewPage = 1,
  feedbackRecord = null,
  busy = false;
const labelName = (value) => names[value] || value;
function element(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = error ? "error" : "";
  $("notice").hidden = false;
}
async function api(path, options = {}) {
  const headers = { ...options.headers };
  if (options.method && options.method !== "GET") {
    headers["Content-Type"] = "application/json";
    headers["X-CSRF-Token"] = sessionInfo?.csrf_token || "";
  }
  const response = await fetch(path, { ...options, headers });
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function call(path, data) {
  return api(path, { method: "POST", body: JSON.stringify(data) });
}
function applySession() {
  $("adminButton").textContent = sessionInfo.is_admin
    ? "退出管理员"
    : "管理员登录";
  $("reviewLocked").hidden = sessionInfo.is_admin;
  $("reviewContent").hidden = !sessionInfo.is_admin;
  $("exportApproved").hidden = !sessionInfo.is_admin;
  $("globalScopeLabel").hidden = !sessionInfo.is_admin;
  if (!sessionInfo.is_admin) $("globalScope").checked = false;
}
async function switchTab(tab) {
  document
    .querySelectorAll("nav button")
    .forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
  document
    .querySelectorAll("main>.panel")
    .forEach((section) => (section.hidden = section.id !== tab));
  try {
    if (tab === "history") await loadHistory();
    if (tab === "stats") await loadStats();
    if (tab === "review" && sessionInfo.is_admin) await loadReviews();
  } catch (error) {
    notice(error.message, true);
  }
}
function date(value) {
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}
function imageCard(url) {
  const card = element("article", undefined, "card");
  const img = element("img");
  img.src = url;
  img.alt = "垃圾识别原图";
  img.loading = "lazy";
  card.append(img);
  const body = element("div", undefined, "card-body");
  card.append(body);
  return { card, body };
}
function feedbackButton(body, recordId, label) {
  const button = element("button", "结果有误？提交纠错", "quiet");
  button.onclick = () => openFeedback(recordId, label);
  body.append(button);
}
function predictionCard(data, filename) {
  const { card, body } = imageCard(`/api/records/${data.record_id}/image`);
  body.append(
    element("h3", labelName(data.class_name)),
    element(
      "span",
      coarse[data.coarse_category] || data.coarse_category,
      "badge",
    ),
    element("p", filename),
  );
  body.append(
    element(
      "p",
      `本地模型置信度 ${(data.confidence * 100).toFixed(1)}% · 耗时 ${(data.cascade_latency_ms || 0).toFixed(0)} ms`,
    ),
  );
  const routing = data.routing || {};
  if (data.uncertain) {
    body.append(
      element(
        "span",
        routing.provider_success ? "云复核已完成" : "建议人工复核",
        "badge warning",
      ),
    );
  }
  body.append(
    element("p", data.advice || "可在识别记录中查看结果或提交纠错。"),
  );
  feedbackButton(body, data.record_id, data.class_name);
  return card;
}
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.readAsDataURL(file);
  });
}
async function classify(files) {
  if (busy) return;
  if (!sessionInfo) {
    notice("会话尚未就绪，请刷新重试", true);
    return;
  }
  if (!files.length) return;
  if (files.length > 16) {
    notice("每批最多选择 16 张图片", true);
    return;
  }
  if (
    files.some(
      (f) =>
        !["image/jpeg", "image/png", "image/webp"].includes(f.type) ||
        f.size > 8 * 1024 * 1024,
    )
  ) {
    notice("仅支持不超过 8 MB 的 JPG、PNG、WEBP 图片", true);
    return;
  }
  busy = true;
  $("predictionResults").replaceChildren();
  $("progress").hidden = false;
  $("imageInput").disabled = true;
  let succeeded = 0;
  try {
    for (let i = 0; i < files.length; i++) {
      $("progress").textContent =
        `正在识别 ${i + 1} / ${files.length}：${files[i].name}`;
      try {
        const image = await readFile(files[i]);
        const result = await call("/predict", {
          image,
          format: "base64",
          filename: files[i].name,
        });
        $("predictionResults").append(predictionCard(result, files[i].name));
        succeeded++;
      } catch (error) {
        const card = element("article", undefined, "card");
        const body = element("div", undefined, "card-body");
        body.append(
          element("h3", files[i].name),
          element("p", error.message, "error"),
        );
        card.append(body);
        $("predictionResults").append(card);
      }
    }
  } finally {
    busy = false;
    $("imageInput").disabled = false;
    $("imageInput").value = "";
    $("progress").textContent =
      `本批完成：${succeeded} 张成功，${files.length - succeeded} 张失败。成功结果已保存。`;
    $("predictionResults").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}
async function loadHistory() {
  const query = new URLSearchParams({
    page: historyPage,
    category: $("categoryFilter").value,
  });
  const data = await api("/api/records?" + query);
  const container = $("historyResults");
  container.replaceChildren();
  if (!data.items.length)
    container.append(
      element("div", "还没有识别记录，先上传一张图片吧。", "empty"),
    );
  for (const item of data.items) {
    const { card, body } = imageCard(item.image_url);
    body.append(
      element("h3", labelName(item.predicted_label)),
      element("span", coarse[item.coarse_category], "badge"),
      element("p", item.filename),
      element("p", date(item.created_at)),
      element("p", `本地置信度 ${(item.confidence * 100).toFixed(1)}%`),
    );
    if (item.feedback_status) {
      body.append(
        element("span", `纠错${statusNames[item.feedback_status]}`, "badge"),
        element("p", `用户建议：${labelName(item.expected_label)}`),
      );
      if (item.review_note)
        body.append(element("p", `审核意见：${item.review_note}`));
    } else feedbackButton(body, item.id, item.predicted_label);
    container.append(card);
  }
  $("historyPage").textContent =
    `第 ${historyPage} / ${Math.max(1, Math.ceil(data.total / data.limit))} 页 · 共 ${data.total} 条`;
  $("historyPrev").disabled = historyPage <= 1;
  $("historyNext").disabled = historyPage * data.limit >= data.total;
}
function openFeedback(id, label) {
  feedbackRecord = id;
  $("feedbackOriginal").textContent = `模型结果：${labelName(label)}`;
  $("expectedLabel").value = label;
  $("feedbackComment").value = "";
  $("feedbackError").textContent = "";
  $("feedbackDialog").showModal();
}
async function loadStats() {
  const data = await api(
    "/api/stats" + ($("globalScope").checked ? "?scope=all" : ""),
  );
  const cards = $("statCards");
  cards.replaceChildren();
  for (const [title, value] of [
    ["识别总次数", data.total],
    ["建议复核次数", data.uncertain],
    ["待审核纠错", data.feedback.pending || 0],
    ["已采纳纠错", data.feedback.approved || 0],
  ]) {
    const card = element("div", undefined, "stat");
    card.append(element("small", title), element("strong", String(value)));
    cards.append(card);
  }
  const pairs = $("correctionPairs");
  pairs.replaceChildren();
  for (const pair of data.correction_pairs) {
    pairs.append(element("li", `${labelName(pair.predicted_label)} → ${labelName(pair.expected_label)}：${pair.n} 次`));
  }
  pairs.hidden = !data.correction_pairs.length;
  $("correctionEmpty").hidden = !!data.correction_pairs.length;
  const chart = $("categoryChart");
  chart.replaceChildren();
  for (const [key, title] of Object.entries(coarse)) {
    const row = element("div", undefined, "chart-row");
    const bar = element("progress", undefined, "chart-bar");
    bar.max = Math.max(1, data.total);
    bar.value = data.categories[key] || 0;
    bar.setAttribute("aria-label", `${title} ${bar.value} 次`);
    row.append(
      element("span", title),
      bar,
      element("span", String(data.categories[key] || 0)),
    );
    chart.append(row);
  }
}
async function loadReviews() {
  const data = await api(
    "/api/reviews?" +
      new URLSearchParams({
        page: reviewPage,
        status: $("reviewStatus").value,
      }),
  );
  const container = $("reviewResults");
  container.replaceChildren();
  if (!data.items.length)
    container.append(element("div", "当前状态下暂无纠错记录。", "empty"));
  for (const item of data.items) {
    const { card, body } = imageCard(`/api/records/${item.record_id}/image`);
    body.append(
      element("h3", `建议：${labelName(item.expected_label)}`),
      element("p", `模型：${labelName(item.predicted_label)}`),
      element("p", `用户说明：${item.comment || "无"}`),
      element("span", statusNames[item.status], "badge"),
    );
    if (item.status === "pending") {
      const note = element("textarea");
      note.placeholder = "填写审核说明（最多 500 字）";
      note.maxLength = 500;
      note.setAttribute("aria-label", "审核说明");
      body.append(note);
      const actions = element("div", undefined, "decision-actions");
      for (const [status, title] of [
        ["approved", "采纳纠错"],
        ["rejected", "驳回建议"],
      ]) {
        const button = element("button", title);
        button.onclick = async () => {
          button.disabled = true;
          try {
            await call(`/api/reviews/${item.id}/decision`, {
              status,
              note: note.value,
              version: item.version,
            });
            notice("审核已保存，原始预测保持可追溯。");
            await loadReviews();
          } catch (error) {
            notice(error.message, true);
            button.disabled = false;
          }
        };
        actions.append(button);
      }
      body.append(actions);
    } else {
      body.append(
        element("p", `审核意见：${item.review_note || "无"}`),
        element("p", date(item.reviewed_at)),
      );
    }
    container.append(card);
  }
  $("reviewPage").textContent =
    `第 ${reviewPage} / ${Math.max(1, Math.ceil(data.total / data.limit))} 页 · 共 ${data.total} 条`;
  $("reviewPrev").disabled = reviewPage <= 1;
  $("reviewNext").disabled = reviewPage * data.limit >= data.total;
}
document
  .querySelectorAll("[data-tab]")
  .forEach((button) => (button.onclick = () => switchTab(button.dataset.tab)));
document
  .querySelectorAll("[data-close]")
  .forEach(
    (button) => (button.onclick = () => $(button.dataset.close).close()),
  );
$("imageInput").onchange = (event) => classify([...event.target.files]);
$("dropzone").onkeydown = (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("imageInput").click();
  }
};
for (const eventName of ["dragover", "dragenter"])
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.add("dragging");
  });
for (const eventName of ["dragleave", "drop"])
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.remove("dragging");
  });
$("dropzone").addEventListener("drop", (event) =>
  classify([...event.dataTransfer.files]),
);
$("categoryFilter").onchange = () => {
  historyPage = 1;
  loadHistory().catch((e) => notice(e.message, true));
};
$("refreshHistory").onclick = () =>
  loadHistory().catch((e) => notice(e.message, true));
$("historyPrev").onclick = () => {
  historyPage--;
  loadHistory().catch((e) => notice(e.message, true));
};
$("historyNext").onclick = () => {
  historyPage++;
  loadHistory().catch((e) => notice(e.message, true));
};
$("reviewStatus").onchange = () => {
  reviewPage = 1;
  loadReviews().catch((e) => notice(e.message, true));
};
$("refreshReviews").onclick = () =>
  loadReviews().catch((e) => notice(e.message, true));
$("reviewPrev").onclick = () => {
  reviewPage--;
  loadReviews().catch((e) => notice(e.message, true));
};
$("reviewNext").onclick = () => {
  reviewPage++;
  loadReviews().catch((e) => notice(e.message, true));
};
$("globalScope").onchange = () =>
  loadStats().catch((e) => notice(e.message, true));
$("adminButton").onclick = async () => {
  if (!sessionInfo) return;
  if (sessionInfo.is_admin) {
    try {
      await call("/api/admin/logout", {});
      sessionInfo.is_admin = false;
      applySession();
      notice("已退出管理员。");
    } catch (error) {
      notice(error.message, true);
    }
  } else {
    $("loginError").textContent = "";
    $("loginDialog").showModal();
  }
};
$("loginForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    await call("/api/admin/login", { token: $("adminToken").value });
    $("adminToken").value = "";
    sessionInfo.is_admin = true;
    applySession();
    $("loginDialog").close();
    await switchTab("review");
  } catch (error) {
    $("loginError").textContent = error.message;
  }
};
$("feedbackForm").onsubmit = async (event) => {
  event.preventDefault();
  try {
    await call(`/api/records/${feedbackRecord}/feedback`, {
      expected_label: $("expectedLabel").value,
      comment: $("feedbackComment").value,
    });
    $("feedbackDialog").close();
    notice("纠错已提交，等待管理员审核。");
    if (!$("history").hidden) await loadHistory();
  } catch (error) {
    $("feedbackError").textContent = error.message;
  }
};
async function init() {
  try {
    sessionInfo = await api("/api/session");
    applySession();
    for (const label of sessionInfo.class_names) {
      const option = element("option", labelName(label));
      option.value = label;
      $("expectedLabel").append(option);
      $("labelScope").append(
        element(
          "span",
          `${labelName(label)} · ${coarse[sessionInfo.class_to_coarse[label]]}`,
        ),
      );
    }
    const health = await api("/health");
    $("connectionDot").className =
      "dot " + (health.model_loaded ? "ready" : "error");
    $("connectionText").textContent = health.model_loaded
      ? "模型已就绪，可以识别"
      : "模型尚未就绪，记录功能可用";
    $("releaseText").textContent =
      `版本 ${health.release?.slice(0, 12) || "development"}`;
  } catch (error) {
    $("connectionDot").className = "dot error";
    $("connectionText").textContent = "服务连接失败";
    notice(error.message, true);
  }
}
init();
