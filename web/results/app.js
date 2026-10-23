"use strict";

// Curated snapshot of the linked experiment reports, dated 2026-10-23.
// Classic scripts and relative asset paths keep file:// viewing fully offline.
const samplingDirectory = "../../results/crust-nuts-conservative/completed-chain-diagnostics";
const regularizationDirectory = "../../results/crust-regularization-scan";
const samplingNotice = "来自已完成的三条链；原四链验收未通过。这些区间不能作为已验证的地壳后验不确定性结论。";
const regularizationNotice = "固定参考中心、界面与误差假设的确定性对照。收敛与低残差不代表结构唯一，也不构成后验可信区间。";
const figures = [
  {id:"structure", category:"sampling", title:"Vs–深度结构与先验 / 样本区间", short:"速度结构与区间", kind:"仅供诊断", file:`${samplingDirectory}/prior-posterior`, notice:samplingNotice,
    description:"左侧为完整深度范围，中间放大浅部，右侧比较样本与先验标准差。灰色为先验 95% 区间，蓝色为样本 95% 区间，黑线为参考模型。",
    alt:"Vs 随深度的先验和三条已完成链的样本区间，以及样本与先验标准差比。"},
  {id:"predictive", category:"sampling", title:"四个模态的频散拟合与标准化残差", short:"频散拟合与残差", kind:"仅供诊断", file:`${samplingDirectory}/predictive`, notice:samplingNotice,
    description:"上排为模态 0–3 的观测、潜在频散曲线与含观测噪声的复制数据区间；下排为按假设 σ 标准化的潜在残差。区间为逐点区间，并非整条曲线的同时覆盖带。",
    alt:"四列分别显示模态 0、1、2、3 的频散拟合，上排含观测和两种 95% 区间，下排为标准化残差。"},
  {id:"cutoff", category:"cutoff", title:"模态截止：半空间条件变化前后的根数", short:"模态截止控制实验", kind:"独立数值证据", file:"../../results/crust-mode-audit-refined/cutoff-continuation",
    notice:"有限区间的独立浮点核查支持模态截止，不能作为任意模型的严格无根证明，也没有使地壳后验通过验收。",
    description:"左右分别为正式期失败模型与预热失败模型。红叉标出原模型，黑色虚线为半空间截止条件；两例跨过条件后，受俘获根数由 2 增至 3。连线仅帮助阅读离散计算点。",
    alt:"正式与预热失败模型在截止条件两侧的受俘获根数；原失败点有两个根，越过临界值的模型有三个根。"},
  {id:"regularization-structure", category:"regularization", title:"不同阻尼权重下的速度结构", short:"权重与速度结构", kind:"数值核对通过", file:`${regularizationDirectory}/structure`, notice:regularizationNotice,
    description:"比较 λ=0、10、30、100、300 下的完整与浅部结构。每组五个初值全部收敛；λ=0 仍有触界与明显初值依赖，非零权重下组内一致而组间结构不同。",
    alt:"五个正则化权重、每组五个初值下的完整深度与浅部 Vs 模型对照。"},
  {id:"lambda-comparison", category:"regularization", title:"阻尼强度、拟合与结构稳定性的关系", short:"权重敏感性总览", kind:"数值核对通过", file:`${regularizationDirectory}/lambda-comparison`, notice:regularizationNotice,
    description:"对照拟合残差、偏离参考的程度和跨权重结构差异。λ=100 是本轮参考设置，没有根据最低残差或最接近 σ 自动选取最优权重。",
    alt:"五个正则化权重下的拟合和结构敏感性比较。"},
  {id:"regularization-residuals", category:"regularization", title:"各阻尼权重的分模态残差", short:"分阶残差对照", kind:"数值核对通过", file:`${regularizationDirectory}/residuals`, notice:regularizationNotice,
    description:"保留不同权重下各模态随频率变化的残差。没有重新标注模态、删除观测或重新估计 σ；总残差随权重增加，不要求各模态残差逐一单调变化。",
    alt:"不同正则化权重下，四个模态的相速度残差随频率变化。"},
  {id:"traces", category:"sampling", title:"已完成三条链的轨迹与混合诊断", short:"采样轨迹", kind:"仅供诊断", file:`${samplingDirectory}/traces`, notice:samplingNotice,
    description:"展示已完成链的指定参数与派生量轨迹。图中诊断只覆盖原链 1、2、3；原链 0 在预热阶段受阻，不能用该子集替代原四链验收。",
    alt:"三条已完成采样链的参数与派生量轨迹，仅用于诊断。"},
  {id:"correlation", category:"sampling", title:"Vs 参数间的样本相关性", short:"参数相关矩阵", kind:"仅供诊断", file:`${samplingDirectory}/correlation`, notice:samplingNotice,
    description:"相关矩阵用于观察已访问区域中的参数耦合。该结果条件于固定层界面、物性关系与参考先验，不代表层数或界面深度的不确定性。",
    alt:"三条已完成链的 Vs 参数相关矩阵，显示参数间的样本耦合。"}
];
const categoryNames = {sampling:"NUTS · 仅供诊断", regularization:"正则化 · 对照完成", cutoff:"模态截止 · 独立核查"};
let activeFilter = "all";
let selectedId = figures[0].id;
const byId = id => document.getElementById(id);
const visibleFigures = () => figures.filter(figure => activeFilter === "all" || figure.category === activeFilter);
const currentFigure = () => figures.find(figure => figure.id === selectedId);
const lightbox = byId("lightbox");

function updateLightbox() {
  const figure = currentFigure();
  byId("lightbox-title").textContent = figure.title;
  byId("lightbox-status").textContent = figure.kind;
  byId("lightbox-status").className = `badge ${figure.category === "sampling" ? "warning" : "success"}`;
  byId("lightbox-notice").textContent = figure.notice;
  byId("lightbox-image").src = `${figure.file}.png`;
  byId("lightbox-image").alt = figure.alt;
  byId("lightbox-pdf").href = `${figure.file}.pdf`;
  byId("lightbox-scroll").classList.remove("actual-size");
  byId("lightbox-scroll").scrollTo(0, 0);
  byId("lightbox-zoom").textContent = "原始尺寸";
  byId("lightbox-zoom").setAttribute("aria-pressed", "false");
  for (const id of ["lightbox-prev", "lightbox-next"]) byId(id).disabled = visibleFigures().length < 2;
}

function renderThumbnails() {
  const container = byId("figure-thumbnails");
  container.replaceChildren();
  for (const figure of visibleFigures()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `thumbnail${figure.id === selectedId ? " active" : ""}`;
    button.setAttribute("aria-pressed", String(figure.id === selectedId));
    button.setAttribute("aria-label", `${figure.short}，${figure.kind}`);
    const img = document.createElement("img");
    img.alt = "";
    img.loading = "lazy";
    img.addEventListener("error", () => { img.hidden = true; });
    img.src = `${figure.file}.png`;
    const label = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = figure.short;
    const detail = document.createElement("small");
    detail.textContent = categoryNames[figure.category];
    label.append(title, detail);
    button.append(img, label);
    button.addEventListener("click", () => selectFigure(figure.id));
    container.append(button);
  }
}

function selectFigure(id) {
  const figure = figures.find(item => item.id === id);
  if (!figure) return;
  selectedId = id;
  byId("figure-title").textContent = figure.title;
  byId("figure-description").textContent = figure.description;
  byId("figure-kind").textContent = figure.kind;
  byId("figure-kind").className = `badge ${figure.category === "sampling" ? "warning" : "success"}`;
  byId("figure-notice").textContent = figure.notice;
  byId("figure-notice").classList.toggle("evidence", figure.category !== "sampling");
  byId("figure-source").textContent = `来源：${figure.file.replace("../../results/", "results/")}.png`;
  byId("figure-png").href = `${figure.file}.png`;
  byId("figure-pdf").href = `${figure.file}.pdf`;
  const image = byId("figure-image");
  byId("image-unavailable").hidden = true;
  byId("figure-expand").hidden = false;
  image.alt = figure.alt;
  image.src = `${figure.file}.png`;
  const visible = visibleFigures();
  byId("figure-count").textContent = `${String(visible.findIndex(item => item.id === id) + 1).padStart(2,"0")} / ${String(visible.length).padStart(2,"0")}`;
  for (const buttonId of ["figure-prev", "figure-next"]) byId(buttonId).disabled = visible.length < 2;
  // Preserve focused thumbnail nodes during selection for keyboard users.
  for (const [index, button] of [...byId("figure-thumbnails").children].entries()) {
    const selected = visible[index].id === id;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  if (lightbox.open) updateLightbox();
}

function moveFigure(direction) {
  const visible = visibleFigures();
  const index = visible.findIndex(item => item.id === selectedId);
  selectFigure(visible[(index + direction + visible.length) % visible.length].id);
}

byId("figure-image").addEventListener("error", () => {
  byId("figure-expand").hidden = true;
  byId("image-unavailable").hidden = false;
});
document.querySelectorAll("[data-filter]").forEach(button => {
  button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(item => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    const visible = visibleFigures();
    if (!visible.some(figure => figure.id === selectedId)) selectedId = visible[0].id;
    renderThumbnails();
    selectFigure(selectedId);
  });
});
byId("figure-prev").addEventListener("click", () => moveFigure(-1));
byId("figure-next").addEventListener("click", () => moveFigure(1));
byId("lightbox-prev").addEventListener("click", () => moveFigure(-1));
byId("lightbox-next").addEventListener("click", () => moveFigure(1));
byId("figure-expand").addEventListener("click", () => {
  updateLightbox();
  lightbox.showModal();
  document.body.style.overflow = "hidden";
  byId("lightbox-close").focus();
});
byId("lightbox-close").addEventListener("click", () => lightbox.close());
lightbox.addEventListener("close", () => { document.body.style.overflow = ""; byId("figure-expand").focus(); });
lightbox.addEventListener("click", event => { if (event.target === lightbox) lightbox.close(); });
lightbox.addEventListener("keydown", event => {
  if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
    event.preventDefault();
    moveFigure(event.key === "ArrowRight" ? 1 : -1);
  }
});
byId("lightbox-zoom").addEventListener("click", event => {
  const zoomed = byId("lightbox-scroll").classList.toggle("actual-size");
  event.currentTarget.setAttribute("aria-pressed", String(zoomed));
  event.currentTarget.textContent = zoomed ? "适应窗口" : "原始尺寸";
});

// Actual computed continuation points, copied from continuations.json. No
// interpolated mode counts or new physical evaluations are produced by this UI.
const cutoffCases = [
  {critical:4.439194324296876, points:[
    {vs:4.427261789312215, roots:[3.071702406223599,3.9393090430316104]},
    {vs:4.438194324296876, roots:[3.0717024062240252,3.9393090430316633]},
    {vs:4.440194324296876, roots:[3.0717024062241522,3.9393090430309883,4.439194324297434]},
    {vs:4.449194324296876, roots:[3.0717024062243397,3.939309043031412,4.439194324297622]}
  ]},
  {critical:4.516440325384723, points:[
    {vs:4.516420303210729, roots:[3.0679313683447536,3.8926934224245677]},
    {vs:4.515440325384723, roots:[3.0679313683440936,3.892693422424928]},
    {vs:4.517440325384723, roots:[3.0679313683446328,3.8926934224259453,4.516440325385477]},
    {vs:4.526440325384723, roots:[3.0679313683448504,3.892693422426145,4.516440325384736]}
  ]}
];
let cutoffIndex = 0;
let pointIndex = 0;
function renderCutoff() {
  const control = cutoffCases[cutoffIndex];
  const point = control.points[pointIndex];
  const offsets = control.points.map(item => item.vs - control.critical);
  const minimum = Math.min(...offsets);
  const maximum = Math.max(...offsets);
  const margin = (maximum - minimum) * .14;
  const lo = minimum - margin, hi = maximum + margin;
  const x = offset => 58 + (offset - lo) / (hi - lo) * 512;
  const y = count => count === 3 ? 65 : 155;
  const ticks = Array.from({length:5}, (_, i) => lo + (hi-lo)*i/4);
  byId("cutoff-chart").innerHTML = `<svg viewBox="0 0 620 235" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <text x="20" y="24">受俘获根数</text>
    ${[2,3].map(count => `<line x1="58" x2="570" y1="${y(count)}" y2="${y(count)}" stroke="#e4eade"/><text x="34" y="${y(count)+4}">${count}</text>`).join("")}
    <line x1="${x(0)}" x2="${x(0)}" y1="35" y2="180" stroke="#8c9c85" stroke-dasharray="4 5"/>
    <text x="${x(0)+8}" y="36">截止条件</text>
    ${ticks.map(tick => `<line x1="${x(tick)}" x2="${x(tick)}" y1="180" y2="184" stroke="#b8c5b1"/><text x="${x(tick)}" y="201" text-anchor="middle">${tick.toFixed(4)}</text>`).join("")}
    <line x1="58" x2="570" y1="180" y2="180" stroke="#d8e2d1"/>
    ${control.points.map((item,i) => `${i === pointIndex ? `<circle cx="${x(offsets[i])}" cy="${y(item.roots.length)}" r="13" fill="none" stroke="${item.roots.length === 3 ? "#458464" : "#b87b4d"}" stroke-opacity=".3" stroke-width="2"/>` : ""}<circle cx="${x(offsets[i])}" cy="${y(item.roots.length)}" r="6" fill="${item.roots.length === 3 ? "#458464" : "#b87b4d"}"/>`).join("")}
    <text x="314" y="228" text-anchor="middle">半空间 Vs − 临界 Vs (km/s)</text></svg>`;
  const available = point.roots.length >= 3;
  byId("cutoff-chart").setAttribute("aria-label", `${cutoffIndex === 0 ? "正式" : "预热"}失败代表模型，截止速度 ${control.critical.toFixed(9)} km/s。原点及临界值下方有 2 个根，上方两个计算点有 3 个根。`);
  byId("point-result").innerHTML = `<strong class="${available ? "present" : ""}">${point.roots.length}<small>个根</small></strong><dl><div><dt>半空间 Vs</dt><dd>${point.vs.toFixed(9)} km/s</dd></div><div><dt>根的相速度</dt><dd>${point.roots.map(root => root.toFixed(6)).join(" / ")}</dd></div></dl><p>${available ? "该计算点具有请求的第三个根。" : "该计算点仅有两个根，审计域内缺少请求的第三个根。"}根的相速度单位为 km/s。</p>`;
  document.querySelectorAll("[data-case]").forEach(button => {
    const selected = Number(button.dataset.case) === cutoffIndex;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  document.querySelectorAll("[data-point]").forEach(button => {
    const selected = Number(button.dataset.point) === pointIndex;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}
document.querySelectorAll("[data-case]").forEach(button => button.addEventListener("click", () => {
  cutoffIndex = Number(button.dataset.case);
  pointIndex = 0;
  renderCutoff();
}));
document.querySelectorAll("[data-point]").forEach(button => button.addEventListener("click", () => {
  pointIndex = Number(button.dataset.point);
  renderCutoff();
}));

const sections = [...document.querySelectorAll("main > section")];
let navigationScheduled = false;
function updateNavigation() {
  const offset = window.innerWidth <= 700 ? 140 : window.innerWidth <= 980 ? 100 : 80;
  const active = [...sections].reverse().find(section => section.getBoundingClientRect().top <= offset) || sections[0];
  document.querySelectorAll("nav a").forEach(link => {
    const selected = link.hash === `#${active.id}`;
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
  navigationScheduled = false;
}
window.addEventListener("scroll", () => {
  if (!navigationScheduled) { navigationScheduled = true; requestAnimationFrame(updateNavigation); }
}, {passive:true});
window.addEventListener("resize", updateNavigation);
renderThumbnails();
selectFigure(selectedId);
renderCutoff();
updateNavigation();
