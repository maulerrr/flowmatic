const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { chromium } = require("playwright");

const ROOT = "C:/Users/BG/Desktop/flowmatic";
const WORKSPACE = path.join(ROOT, "outputs/019ebb43-9f9d-7692-977f-67df56a953ba/presentations/flowmatic-defence");
const OUTPUT = path.join(WORKSPACE, "output");
const ASSETS = path.join(WORKSPACE, "assets");
fs.mkdirSync(OUTPUT, { recursive: true });
fs.mkdirSync(ASSETS, { recursive: true });

const OUT = {
  pptx: path.join(OUTPUT, "Flowmatic_Master_Thesis_Defence_Final.pptx"),
  pdf: path.join(OUTPUT, "Flowmatic_Master_Thesis_Defence_Final.pdf"),
  plan: path.join(OUTPUT, "Flowmatic_Defence_Slide_Plan.md"),
  audit: path.join(OUTPUT, "Flowmatic_Presentation_Evidence_Audit.md"),
  script: path.join(OUTPUT, "Flowmatic_Defence_Script.md"),
  qa: path.join(OUTPUT, "Flowmatic_Presentation_QA.md"),
  html: path.join(OUTPUT, "Flowmatic_Master_Thesis_Defence_Final.html"),
};

const AITU_LOGO = path.join(ROOT, "thesis/Flowmatic-Thesis-v3/logos/AITU.png");
const screenshots = [
  path.join(ROOT, "thesis/Flowmatic-Thesis-v3/figures/screenshots/02-upload.png"),
  path.join(ROOT, "thesis/Flowmatic-Thesis-v3/figures/screenshots/08-smart-city-live-workbench.png"),
  path.join(ROOT, "thesis/Flowmatic-Thesis-v3/figures/screenshots/10-workflow-graph.png"),
].filter(fs.existsSync);

const C = {
  ink: "111827",
  muted: "5B6575",
  blue: "0B3D91",
  cyan: "08A6C9",
  green: "168A5A",
  amber: "D08312",
  red: "B42318",
  paper: "F7FAFC",
  line: "CBD5E1",
  paleBlue: "EAF3FF",
  paleGreen: "E9F7EF",
  paleAmber: "FFF4DF",
  paleRed: "FDECEC",
  white: "FFFFFF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Ramazan Seiitbek";
pptx.company = "Astana IT University";
pptx.subject = "Flowmatic master's thesis defence";
pptx.title = "Flowmatic Master Thesis Defence";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};
pptx.defineLayout({ name: "OFFICIAL_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "OFFICIAL_WIDE";
pptx.margin = 0;

function addBackground(slide, opts = {}) {
  const dark = opts.dark || false;
  slide.background = { color: dark ? "0B1220" : C.paper };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.16,
    fill: { color: dark ? C.cyan : C.blue },
    line: { color: dark ? C.cyan : C.blue, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 7.32,
    w: 13.333,
    h: 0.18,
    fill: { color: dark ? "162033" : "E6EEF8" },
    line: { color: dark ? "162033" : "E6EEF8", transparency: 100 },
  });
}

function addHeader(slide, n, kicker, dark = false) {
  const tc = dark ? C.white : C.ink;
  if (fs.existsSync(AITU_LOGO)) slide.addImage({ path: AITU_LOGO, x: 0.42, y: 0.32, w: 0.52, h: 0.52 });
  slide.addText("ASTANA IT UNIVERSITY", { x: 1.04, y: 0.37, w: 3.4, h: 0.18, fontSize: 8.5, color: dark ? "DDEBFF" : C.blue, bold: true, margin: 0 });
  slide.addText(kicker.toUpperCase(), { x: 0.42, y: 0.92, w: 2.5, h: 0.22, fontSize: 8.5, color: dark ? C.cyan : C.blue, bold: true, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 0.42, y: 1.2, w: 12.48, h: 0, line: { color: dark ? "2B3B56" : C.line, pt: 0.8 } });
  slide.addText(String(n).padStart(2, "0"), { x: 12.18, y: 7.34, w: 0.5, h: 0.12, fontSize: 8, color: dark ? "B7C4D8" : C.muted, align: "right", margin: 0 });
  slide.addText("Flowmatic master's thesis defence", { x: 0.42, y: 7.34, w: 3.3, h: 0.12, fontSize: 7.5, color: dark ? "B7C4D8" : C.muted, margin: 0 });
  slide._dark = dark;
  slide._tc = tc;
}

function title(slide, text, y = 1.32, h = 0.65, size = 26) {
  slide.addText(text, { x: 0.42, y, w: 12.4, h, fontFace: "Aptos Display", fontSize: size, bold: true, color: slide._tc || C.ink, margin: 0.02, breakLine: false });
}

function subtitle(slide, text, y, h = 0.36) {
  slide.addText(text, { x: 0.44, y, w: 11.9, h, fontSize: 13.2, color: slide._dark ? "CBD5E1" : C.muted, margin: 0.02, fit: "shrink" });
}

function foot(slide, text, dark = false) {
  slide.addText(text, { x: 8.1, y: 7.34, w: 4.3, h: 0.12, fontSize: 7.2, color: dark ? "B7C4D8" : C.muted, align: "right", margin: 0 });
}

function box(slide, x, y, w, h, text, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.06,
    fill: { color: opts.fill || C.white },
    line: { color: opts.line || C.line, pt: opts.pt || 0.7 },
  });
  slide.addText(text, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: h - 0.16,
    fontSize: opts.size || 11.5,
    bold: opts.bold || false,
    color: opts.color || C.ink,
    margin: 0.03,
    valign: "mid",
    fit: "shrink",
    breakLine: false,
  });
}

function metric(slide, x, y, w, value, label, note, color = C.blue) {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.92, fill: { color: C.white }, line: { color, pt: 1.1 } });
  slide.addText(value, { x: x + 0.12, y: y + 0.12, w: w - 0.24, h: 0.28, fontSize: 21, bold: true, color, margin: 0 });
  slide.addText(label, { x: x + 0.13, y: y + 0.47, w: w - 0.26, h: 0.19, fontSize: 8.8, bold: true, color: C.ink, margin: 0 });
  slide.addText(note, { x: x + 0.13, y: y + 0.68, w: w - 0.26, h: 0.17, fontSize: 7.6, color: C.muted, margin: 0, fit: "shrink" });
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  for (const item of items) runs.push({ text: item, options: { bullet: { indent: 12 }, hanging: 4, breakLine: true } });
  slide.addText(runs, { x, y, w, h, fontSize: opts.size || 12.5, color: opts.color || (slide._dark ? "E5EDF8" : C.ink), margin: 0.02, fit: "shrink", breakLine: false, paraSpaceAfterPt: 5 });
}

function flow(slide, nodes, x, y, w, opts = {}) {
  const gap = opts.gap || 0.12;
  const nodeW = (w - gap * (nodes.length - 1)) / nodes.length;
  nodes.forEach((node, i) => {
    const nx = x + i * (nodeW + gap);
    box(slide, nx, y, nodeW, opts.h || 0.58, node, { fill: opts.fill || C.white, line: opts.line || C.blue, size: opts.size || 9.5, bold: true, color: opts.color || C.ink });
    if (i < nodes.length - 1) {
      slide.addShape(pptx.ShapeType.line, { x: nx + nodeW + 0.02, y: y + (opts.h || 0.58) / 2, w: gap - 0.04, h: 0, line: { color: opts.arrow || C.blue, pt: 1.1, beginArrowType: "none", endArrowType: "triangle" } });
    }
  });
}

function barChart(slide, x, y, w, h, rows, opts = {}) {
  const max = Math.max(...rows.flatMap(r => [r.before ?? 0, r.after ?? 0, r.value ?? 0]));
  const rowH = h / rows.length;
  rows.forEach((r, i) => {
    const yy = y + i * rowH + 0.04;
    slide.addText(r.label, { x, y: yy, w: 1.25, h: 0.2, fontSize: 8.2, color: C.muted, margin: 0 });
    if (r.before !== undefined) {
      slide.addShape(pptx.ShapeType.rect, { x: x + 1.38, y: yy, w: (w - 2.1) * (r.before / max), h: 0.12, fill: { color: opts.before || C.red }, line: { color: opts.before || C.red, transparency: 100 } });
      slide.addShape(pptx.ShapeType.rect, { x: x + 1.38, y: yy + 0.16, w: (w - 2.1) * (r.after / max), h: 0.12, fill: { color: opts.after || C.green }, line: { color: opts.after || C.green, transparency: 100 } });
      slide.addText(`${r.before.toFixed(3)} -> ${r.after.toFixed(3)}`, { x: x + w - 0.62, y: yy + 0.01, w: 0.72, h: 0.18, fontSize: 7.2, color: C.ink, margin: 0 });
    } else {
      slide.addShape(pptx.ShapeType.rect, { x: x + 1.38, y: yy + 0.08, w: (w - 2.1) * (r.value / max), h: 0.16, fill: { color: r.color || C.blue }, line: { color: r.color || C.blue, transparency: 100 } });
      slide.addText(r.text || String(r.value), { x: x + w - 0.72, y: yy + 0.07, w: 0.82, h: 0.16, fontSize: 7.5, color: C.ink, margin: 0 });
    }
  });
}

function sectionSlide(n, kicker, big, small, dark = true) {
  const slide = pptx.addSlide();
  addBackground(slide, { dark });
  addHeader(slide, n, kicker, dark);
  slide.addText(big, { x: 0.8, y: 2.05, w: 11.8, h: 1.2, fontFace: "Aptos Display", fontSize: 38, bold: true, color: dark ? C.white : C.ink, margin: 0.02, fit: "shrink" });
  slide.addText(small, { x: 0.82, y: 3.55, w: 9.8, h: 0.65, fontSize: 15, color: dark ? "D5E4F6" : C.muted, margin: 0.02, fit: "shrink" });
  return slide;
}

const slides = [];
function addSlide(spec, draw) {
  const slide = pptx.addSlide();
  addBackground(slide, { dark: spec.dark });
  addHeader(slide, spec.n, spec.kicker, spec.dark);
  title(slide, spec.title, spec.titleY || 1.31, spec.titleH || 0.56, spec.titleSize || 25);
  if (spec.sub) subtitle(slide, spec.sub, spec.subY || 1.92, spec.subH || 0.36);
  draw(slide);
  if (spec.source) foot(slide, spec.source, spec.dark);
  if (spec.notes) slide.addNotes(spec.notes);
  slides.push(spec);
  return slide;
}

const commonNotes = {
  vulnerable: "Answer structure for vulnerable results: direct answer, evidence, scope boundary, next validation step.",
};

addSlide({
  n: 1, kicker: "Title", title: "Development of an Intelligent Assistant for Data Preparation Automation in Urban Transportation Management Systems", titleSize: 24, titleH: 1.1,
  sub: "Flowmatic - master's thesis defence | 7M06105 Computer Science and Engineering",
  source: "Thesis v3 title page; official AITU template",
  notes: "Takeaway: Flowmatic is presented as an evaluated research prototype, not as a production-certified municipal system.\nExplain the title, candidate, programme, supervisor, and year. Transition: the defence begins with the operational problem that motivates the platform.",
}, s => {
  box(s, 0.72, 2.55, 5.8, 1.08, "Candidate: Ramazan Seiitbek\nSupervisor: Aivar Sakhipov\nSchool of Software Engineering, Astana IT University", { fill: C.white, line: C.blue, bold: true, size: 14 });
  box(s, 7.0, 2.55, 4.95, 1.08, "Astana, Kazakhstan\nJune 2026", { fill: C.paleBlue, line: C.cyan, bold: true, size: 16, color: C.blue });
  flow(s, ["Problem", "Method", "Implementation", "Evidence", "Limits"], 0.74, 4.65, 11.8, { h: 0.62, fill: "F8FBFF", line: C.blue, size: 10.5 });
});

addSlide({
  n: 2, kicker: "Relevance", title: "Dirty transport data breaks the path from operations to analytics.",
  sub: "The research problem is not just missing values; it is the missing auditable preparation-to-inference chain.",
  source: "Thesis v3 Sec. 1.1; recommendations: problem -> method -> result",
  notes: "Takeaway: Flowmatic addresses the preparation chain, not generic traffic prediction. Explain heterogeneous sources, defects, and the downstream consequence. Limitation: no external accident statistics are used because the thesis evidence is about data preparation and analytics reliability. Transition to the gap.",
}, s => {
  flow(s, ["CSV exports", "API feeds", "WebSocket telemetry", "Sensor simulators"], 0.55, 2.38, 5.8, { h: 0.52, fill: C.white, line: C.line, size: 8.8 });
  flow(s, ["Missing", "Duplicates", "Invalid GPS", "Stale time"], 0.55, 3.35, 5.8, { h: 0.52, fill: C.paleRed, line: C.red, size: 8.8, arrow: C.red });
  flow(s, ["Forecasting", "Anomaly detection", "Imputation", "Routing"], 0.55, 4.32, 5.8, { h: 0.52, fill: C.paleAmber, line: C.amber, size: 8.8, arrow: C.amber });
  box(s, 7.0, 2.35, 4.95, 2.55, "Research problem\n\nUrban-transport preparation is often performed outside the model workflow, leaving weak lineage, inconsistent quality measurement, and unclear downstream impact.", { fill: C.white, line: C.blue, size: 17, bold: false });
});

addSlide({
  n: 3, kicker: "Gap", title: "The gap is measurement linkage, not a claim that no tools exist.",
  sub: "Existing systems solve parts of the workflow; the thesis evaluates the chain end to end within a prototype.",
  source: "Thesis v3 Sec. 2.12-2.17; review Q1 risk S03/S07",
  notes: "Takeaway: the novelty is framed against nearby systems without pretending they do not exist. Explain the four columns. This pre-empts 'why not pandas, Airflow, MLflow, or AutoML'. Transition to aim and research questions.",
}, s => {
  const headers = ["Existing solutions", "Typical limit", "Research gap", "Flowmatic contribution"];
  headers.forEach((h, i) => box(s, 0.52 + i * 3.12, 2.25, 2.8, 0.52, h, { fill: C.blue, line: C.blue, color: C.white, bold: true, size: 10 }));
  const rows = [
    ["Scripts / notebooks", "Hard to audit and replay", "No shared evidence trail", "DQI + cleaning + export records"],
    ["ETL / MLOps tools", "Model and prep tracked apart", "Weak transport-specific validation", "Preparation tied to model registry"],
    ["ITS analytics", "Models evaluated separately", "Unclear impact of data defects", "Stress tests + ablation + routing check"],
  ];
  rows.forEach((r, ri) => r.forEach((txt, ci) => box(s, 0.52 + ci * 3.12, 2.92 + ri * 0.78, 2.8, 0.55, txt, { fill: ci === 3 ? C.paleGreen : C.white, line: C.line, size: 8.7 })));
  box(s, 0.8, 5.55, 11.6, 0.58, "Contribution boundary: Flowmatic does not claim a new universal neural architecture, new DQI theory, or production-certified ITS deployment.", { fill: C.paleAmber, line: C.amber, size: 13, bold: true, color: C.ink });
});

addSlide({
  n: 4, kicker: "Aim", title: "The aim is implemented through six objectives and four scoped questions.",
  sub: "Each research question is tied to measurable evidence and a visible limitation.",
  source: "Thesis v3 Sec. 1.2, 3.3, 5.9",
  notes: "Takeaway: the thesis is research because objectives are tested through measurable protocols. State aim in one sentence. Explain object and subject. Transition to novelty boundaries.",
}, s => {
  box(s, 0.55, 2.15, 4.1, 1.16, "Aim\nDevelop and evaluate an intelligent software platform for automated transport-data preparation with DQI, reproducible preprocessing, and downstream model integration.", { fill: C.white, line: C.blue, size: 12.5, bold: true });
  box(s, 4.92, 2.15, 3.0, 1.16, "Object\nData preparation process in urban transportation management systems.", { fill: C.paleBlue, line: C.cyan, size: 12.2, bold: true });
  box(s, 8.22, 2.15, 3.65, 1.16, "Subject\nSoftware mechanisms for quality assessment, preprocessing, routing, and model-ready data preparation.", { fill: C.paleBlue, line: C.cyan, size: 11.4, bold: true });
  const rqs = [
    ["RQ1", "DQI recovery under corruption"],
    ["RQ2", "Batch and streaming prototype operation"],
    ["RQ3", "Model portfolio vs simple baselines"],
    ["RQ4", "Rule-router policy conformance"],
  ];
  rqs.forEach((r, i) => {
    box(s, 0.7 + i * 3.05, 4.22, 2.65, 0.9, `${r[0]}\n${r[1]}`, { fill: i % 2 ? C.white : C.paleGreen, line: C.line, size: 10.6, bold: true });
  });
});

addSlide({
  n: 5, kicker: "Novelty", title: "The scientific claim is the evaluated chain, not the existence of the platform.",
  sub: "Prior publications and standard components are separated from thesis-new evidence.",
  source: "Thesis v3 Sec. 1.3, 2.13.1, Appendix A; review W1",
  notes: "Takeaway: this slide answers 'what is new compared with your prior publication'. Explain that platform concepts are continued, while the defence rests on new evaluation: stress tests, baselines, ablation, routing conformance, and reproducibility. Transition to method.",
}, s => {
  box(s, 0.55, 2.15, 3.75, 2.5, "Scientific contribution\nLinked evaluation protocol: DQI recovery, downstream-model effect, routing conformance, and reproducibility artifacts.", { fill: C.paleGreen, line: C.green, size: 13, bold: true });
  box(s, 4.62, 2.15, 3.75, 2.5, "Engineering contribution\nContainerised Flowmatic prototype: Vue, NestJS, PostgreSQL, MinIO, RabbitMQ, simulators, inference service.", { fill: C.paleBlue, line: C.blue, size: 13, bold: true });
  box(s, 8.68, 2.15, 3.75, 2.5, "Thesis-new evidence\nCorruption stress test, baseline tables, preparation ablation, 140-event routing check, explicit limitation ledger.", { fill: C.paleAmber, line: C.amber, size: 13, bold: true });
  box(s, 1.1, 5.45, 10.95, 0.5, "Not claimed: new neural architecture, learned router, universal DQI theory, or live municipal production deployment.", { fill: C.white, line: C.red, size: 13.5, bold: true, color: C.red });
});

addSlide({
  n: 6, kicker: "Methodology", title: "The research workflow tests the artefact under controlled, reproducible conditions.",
  source: "Thesis v3 Ch. 3, Ch. 5; recommendations: research pipeline",
  notes: "Takeaway: design-build-evaluate is made scientific through testable protocols. Walk from literature to artifact to stress tests, baselines, ablations, and reproducibility. Transition to data scope.",
}, s => {
  flow(s, ["Literature + gap", "Artifact design", "Implementation", "Dataset prep"], 0.6, 2.15, 11.9, { h: 0.62, fill: C.white, line: C.blue, size: 11 });
  flow(s, ["Corruption injection", "Temporal splits", "Baselines", "Ablations"], 0.6, 3.35, 11.9, { h: 0.62, fill: C.paleGreen, line: C.green, size: 11, arrow: C.green });
  flow(s, ["Routing replay", "Latency checks", "Hugging Face artifacts", "Threats to validity"], 0.6, 4.55, 11.9, { h: 0.62, fill: C.paleAmber, line: C.amber, size: 11, arrow: C.amber });
  subtitle(s, "Falsifiable centre: if preparation does not improve quality or downstream behavior under controlled defects, the main thesis claim weakens.", 5.82, 0.38);
});

addSlide({
  n: 7, kicker: "Data Scope", title: "The evidence is reproducible, but not all data are externally real-world validated.",
  sub: "This slide makes semi-synthetic and simulated data visible before results appear.",
  source: "Thesis v3 Sec. 3.4, 5.2.1, Appendix C",
  notes: "Takeaway: trust comes from transparent provenance, not pretending the data are fully public municipal logs. Explain Astana semi-synthetic, HF datasets, simulated feeds, injected anomalies, and rule labels. Transition to DQI method.",
}, s => {
  const rows = [
    ["Astana traffic", "NDA-derived semi-synthetic", "30,000 rows", "Batch API, stress, forecasting", "Reproducible CSV; source calibration limited"],
    ["HF weather / ETT / traffic", "Public bundles", "Offline windows", "Neural training", "Not all re-uploaded through Nest API"],
    ["Simulator events", "Generated HTTP/WebSocket", "140 routing events", "Auto routing check", "Policy-aligned labels"],
    ["Injected defects/anomalies", "Controlled perturbation", "0-20% corruption", "Stress / TranAD AUROC", "Not municipal incident labels"],
  ];
  const cols = [0.55, 2.75, 5.05, 6.65, 8.75];
  ["Dataset", "Provenance", "Scale", "Use", "Limit"].forEach((h, i) => box(s, cols[i], 2.12, i === 4 ? 3.55 : 1.95, 0.44, h, { fill: C.blue, line: C.blue, color: C.white, bold: true, size: 9 }));
  rows.forEach((r, ri) => r.forEach((txt, ci) => box(s, cols[ci], 2.68 + ri * 0.78, ci === 4 ? 3.55 : 1.95, 0.56, txt, { fill: ri % 2 ? C.white : "F9FBFD", line: C.line, size: 7.8 })));
});

addSlide({
  n: 8, kicker: "Preparation", title: "DQI makes preparation auditable before data reach models.",
  sub: "Six dimensions are operationalised through profiling, rules, cleaning, and export records.",
  source: "Thesis v3 Sec. 3.6-3.7; quality.service.ts; cleaning.service.ts",
  notes: "Takeaway: the DQI is an operational adaptation of known data-quality dimensions. Explain six dimensions and the raw->profile->DQI->repair/reject->prepared->audit flow. Transition to architecture.",
}, s => {
  flow(s, ["Raw input", "Profiling", "Six-D DQI", "Repair / reject", "Prepared data", "Audit artifact"], 0.55, 2.15, 12.0, { h: 0.58, fill: C.white, line: C.blue, size: 9.5 });
  ["Completeness", "Consistency", "Accuracy", "Timeliness", "Uniqueness", "Validity"].forEach((d, i) => {
    box(s, 0.7 + (i % 3) * 3.95, 3.35 + Math.floor(i / 3) * 0.88, 3.35, 0.54, d, { fill: [C.paleBlue, C.paleGreen, C.paleAmber][i % 3], line: C.line, bold: true, size: 12 });
  });
  box(s, 1.0, 5.45, 11.3, 0.5, "Important boundary: at 20% corruption, quality recovery includes row rejection; it is not purely in-place imputation.", { fill: C.white, line: C.amber, size: 13, bold: true, color: C.amber });
});

addSlide({
  n: 9, kicker: "Architecture", title: "Flowmatic connects auditable preparation with task-specific inference.",
  source: "docker-compose.yml; Prisma schema; backend smart-city/model registry modules",
  notes: "Takeaway: this is a real multi-service prototype. Walk left to right: frontend, backend modules, queue/storage/db, simulator, inference, registry. Clarify optional LLM and federated demo are not central evaluated claims. Transition to intelligence definition.",
}, s => {
  box(s, 0.55, 2.1, 2.1, 1.0, "Vue frontend\nUpload + smart-city workbench", { fill: C.paleBlue, line: C.blue, size: 10.5, bold: true });
  box(s, 3.05, 2.1, 2.4, 1.0, "NestJS API\nAuth, ingestion, quality, cleaning, export", { fill: C.white, line: C.blue, size: 10.1, bold: true });
  box(s, 5.85, 1.72, 2.0, 0.72, "PostgreSQL\nmetadata + tenancy", { fill: C.paleGreen, line: C.green, size: 9.3, bold: true });
  box(s, 5.85, 2.65, 2.0, 0.72, "MinIO/S3\nraw + cleaned files", { fill: C.paleGreen, line: C.green, size: 9.3, bold: true });
  box(s, 5.85, 3.58, 2.0, 0.72, "RabbitMQ\nasync jobs", { fill: C.paleGreen, line: C.green, size: 9.3, bold: true });
  box(s, 8.25, 2.1, 2.0, 1.0, "Model registry\nlocal + HF metadata", { fill: C.paleAmber, line: C.amber, size: 10.5, bold: true });
  box(s, 10.65, 2.1, 2.1, 1.0, "Inference service\nTorchScript/Python API", { fill: C.paleAmber, line: C.amber, size: 10.5, bold: true });
  ["0.55,2.6,2.5", "2.65,2.6,0.38", "5.45,2.6,0.38", "7.85,2.6,0.38", "10.25,2.6,0.38"].forEach(str => {
    const [x, y, w] = str.split(",").map(Number);
    s.addShape(pptx.ShapeType.line, { x, y, w, h: 0, line: { color: C.blue, pt: 1.1, endArrowType: "triangle" } });
  });
  box(s, 1.2, 5.35, 10.7, 0.52, "Optional / not central evaluation: LLM narration and federated coordinator demo. They are prototype features, not the basis of the scientific claim.", { fill: C.white, line: C.red, size: 12.5, bold: true, color: C.red });
});

addSlide({
  n: 10, kicker: "Assistant", title: "The system is intelligent as a bounded hybrid assistant, not as general AI.",
  sub: "Automation combines deterministic diagnostics, policy routing, learned models, and operator control.",
  source: "pipeline-model-router.service.ts; model registry; thesis v3 Sec. 3.11",
  notes: "Takeaway: answer 'why intelligent assistant?' precisely. Deterministic checks and rules are legitimate automation; learned components are task-specific. Operator remains in control. Transition to experimental protocol.",
}, s => {
  flow(s, ["Profiling", "DQI rules", "Preparation decisions", "Payload profile", "Policy routing", "Learned inference", "Operator output"], 0.45, 2.25, 12.3, { h: 0.6, fill: C.white, line: C.blue, size: 8.4 });
  box(s, 0.8, 3.65, 3.45, 1.05, "Deterministic automation\nschema checks, missing values, duplicates, ranges, stale records", { fill: C.paleBlue, line: C.blue, size: 11.2, bold: true });
  box(s, 4.85, 3.65, 3.45, 1.05, "Policy decisions\nManual or Auto routing through metadata and priority rules", { fill: C.paleAmber, line: C.amber, size: 11.2, bold: true });
  box(s, 8.9, 3.65, 3.45, 1.05, "Task-specific learning\nforecasting, anomaly, imputation, severity classification", { fill: C.paleGreen, line: C.green, size: 11.2, bold: true });
  subtitle(s, "LLM narration is optional and not used as the scientific basis of the thesis.", 5.55);
});

addSlide({
  n: 11, kicker: "Protocol", title: "The experiment design separates leakage control from statistical strength.",
  sub: "Temporal hold-out is strong; three-seed variability should be read descriptively.",
  source: "Thesis v3 Sec. 5.2.2; review W3",
  notes: "Takeaway: the split prevents direct row leakage, but n=3 is limited. Say values are multi-seed descriptive variability, not broad statistical significance. Transition to stress test results.",
}, s => {
  flow(s, ["70% train", "15% validation", "15% test"], 0.85, 2.35, 6.2, { h: 0.72, fill: C.white, line: C.blue, size: 13 });
  box(s, 7.55, 2.12, 4.35, 1.1, "Leakage controls\nTemporal split, scalers fitted on train, fixed test tail, identical partition for baselines.", { fill: C.paleGreen, line: C.green, size: 12.2, bold: true });
  box(s, 1.0, 4.18, 3.2, 1.05, "Seeds\n42, 7, 2026", { fill: C.white, line: C.line, size: 18, bold: true, color: C.blue });
  box(s, 4.6, 4.18, 3.2, 1.05, "Metrics\nRMSE, macro-F1, AUROC, masked MSE", { fill: C.white, line: C.line, size: 12.5, bold: true });
  box(s, 8.2, 4.18, 3.2, 1.05, "Caution\nNo broad significance claim from n=3", { fill: C.paleAmber, line: C.amber, size: 12.5, bold: true });
});

addSlide({
  n: 12, kicker: "RQ1", title: "Preparation recovered measurable quality, partly by rejecting invalid records.",
  sub: "At 20% corruption, DQI improved from 0.788 to 0.833 while 987 rows were dropped.",
  source: "thesis_v2_evidence.json; generated_tables.tex",
  notes: "Takeaway: the best preparation evidence is the stress test, not the clean 100/100 upload. Explain red/green bars and row drops. Scope boundary: row rejection can remove potentially useful rare records. Transition to model evidence.",
}, s => {
  barChart(s, 0.75, 2.28, 7.05, 2.55, [
    { label: "0%", before: 0.833, after: 0.833 },
    { label: "1%", before: 0.831, after: 0.833 },
    { label: "5%", before: 0.823, after: 0.833 },
    { label: "10%", before: 0.811, after: 0.833 },
    { label: "20%", before: 0.788, after: 0.833 },
  ]);
  metric(s, 8.35, 2.2, 1.45, "0.788", "Before", "20% corruption", C.red);
  metric(s, 10.05, 2.2, 1.45, "0.833", "After", "cleaning", C.green);
  metric(s, 8.35, 3.5, 3.15, "987", "Rows dropped", "9.87% of 10,000", C.amber);
  box(s, 8.35, 4.85, 3.15, 0.82, "Interpretation\nQuality recovery is measurable, but the mechanism is auditably conservative: reject or repair defective records.", { fill: C.white, line: C.line, size: 11.5, bold: true });
});

addSlide({
  n: 13, kicker: "RQ3", title: "Downstream results support capability, with explicit task limits.",
  sub: "Metrics are separated by task to avoid pretending RMSE, F1, AUROC, and MSE are comparable.",
  source: "multi_seed_aggregate.md; generated_baselines.tex; thesis_v2_evidence.json",
  notes: "Takeaway: the portfolio is evidence for integrated task slots, not SOTA benchmarking. Mention key comparators and caveats: rule labels, injected anomalies, z-score windows. Transition to ablation.",
}, s => {
  const rows = [
    ["Forecast", "PatchTST density", "0.561 RMSE", "vs naive z-RMSE 1.063"],
    ["Classification", "Transformer severity", "0.911 macro-F1", "rule-based labels"],
    ["Anomaly", "TranAD", "0.841 AUROC", "injected point anomalies"],
    ["Imputation", "SAITS", "0.640 masked MSE", "vs mean baseline 1.000"],
  ];
  ["Task", "Model", "Headline", "Scope note"].forEach((h, i) => box(s, 0.65 + i * 3.05, 2.18, 2.7, 0.45, h, { fill: C.blue, line: C.blue, color: C.white, bold: true, size: 9 }));
  rows.forEach((r, ri) => r.forEach((txt, ci) => box(s, 0.65 + ci * 3.05, 2.78 + ri * 0.72, 2.7, 0.48, txt, { fill: ri % 2 ? C.white : "F9FBFD", line: C.line, size: 8.6, bold: ci === 2 })));
  box(s, 0.95, 5.85, 11.3, 0.45, "Defence framing: the models certify task slots inside Flowmatic; they do not establish a new forecasting or classification algorithm.", { fill: C.paleAmber, line: C.amber, size: 12.2, bold: true });
});

addSlide({
  n: 14, kicker: "Ablation", title: "Preparation improved forecasting under a controlled upper-bound experiment.",
  sub: "PatchTST RMSE dropped from 0.9045 to 0.5802 after oracle-style restoration on 20% corrupted data.",
  source: "thesis_v2_evidence.json; generated_prep_ablation.tex",
  notes: "Takeaway: this is the strongest preparation-to-downstream linkage, but it is not deployable imputation performance. State 'oracle restoration upper-bound' before the examiner does. Transition to implementation/performance.",
}, s => {
  barChart(s, 0.9, 2.35, 6.4, 1.5, [
    { label: "Prep off", value: 0.9045, text: "0.9045", color: C.red },
    { label: "Prep on", value: 0.5802, text: "0.5802", color: C.green },
  ]);
  metric(s, 7.9, 2.2, 1.7, "0.3243", "RMSE delta", "lower is better", C.green);
  metric(s, 9.95, 2.2, 1.7, "20%", "Corruption", "controlled input", C.amber);
  box(s, 1.0, 4.45, 5.4, 0.88, "What it proves\nRestoring lost information improves downstream prediction under controlled corruption.", { fill: C.paleGreen, line: C.green, size: 12.8, bold: true });
  box(s, 6.9, 4.45, 5.4, 0.88, "What it does not prove\nA deployable imputation method will recover the same amount on live municipal data.", { fill: C.paleRed, line: C.red, size: 12.8, bold: true });
});

addSlide({
  n: 15, kicker: "Implementation", title: "The prototype works, but the measured boundary matters.",
  sub: "Use core processing and inference-latency numbers without turning them into production SLA claims.",
  source: "UPLOAD_PIPELINE_REPORT.md; streaming_benchmark.csv; thesis screenshots",
  notes: "Takeaway: Flowmatic is real and measurable, but not production certified. Explain the timing boundaries: 406 ms batch processing, TorchScript latency from benchmark, not full wall-clock SLA. Transition to RQ closure.",
}, s => {
  metric(s, 0.65, 2.1, 2.05, "406 ms", "Batch processing", "30,000-row Astana export", C.blue);
  metric(s, 3.0, 2.1, 2.05, "<2 ms", "Compact inference", "single-event benchmark", C.green);
  metric(s, 5.35, 2.1, 2.05, "Docker", "Prototype stack", "local Compose topology", C.amber);
  screenshots.slice(0, 3).forEach((img, i) => s.addImage({ path: img, x: 0.7 + i * 4.05, y: 3.35, w: 3.55, h: 1.85 }));
  box(s, 8.0, 2.1, 4.2, 0.92, "Not claimed\nNo HA, SLA, penetration test, formal municipal deployment, or concurrency p95/p99 study.", { fill: C.paleRed, line: C.red, size: 11.6, bold: true });
});

addSlide({
  n: 16, kicker: "Closure", title: "Each research question has evidence and an explicit scope limit.",
  source: "Thesis v3 Table 5.19 / Sec. 5.9",
  notes: "Takeaway: this is the defence consolidation slide. Walk each row quickly: supported, evidence, scope limit. Transition to personal contribution and publications.",
}, s => {
  const rows = [
    ["RQ1", "DQI before/after stress test", "Supported under corruption", "clean files show ceiling effect"],
    ["RQ2", "406 ms batch + smart-city demo", "Supported for prototype", "Astana API validation only"],
    ["RQ3", "baselines, portfolio, ablations", "Supported by task", "limited baselines / n=3"],
    ["RQ4", "140-event routing replay", "Supported as conformance", "not expert-labelled accuracy"],
  ];
  ["RQ", "Evidence", "Outcome", "Scope limit"].forEach((h, i) => box(s, 0.7 + [0,1.4,5.0,8.2][i], 2.15, [1.0,3.2,2.75,3.25][i], 0.45, h, { fill: C.blue, line: C.blue, color: C.white, bold: true, size: 9 }));
  rows.forEach((r, ri) => r.forEach((txt, ci) => box(s, 0.7 + [0,1.4,5.0,8.2][ci], 2.78 + ri * 0.74, [1.0,3.2,2.75,3.25][ci], 0.48, txt, { fill: ri % 2 ? C.white : "F9FBFD", line: C.line, size: 8.6, bold: ci === 2 })));
});

addSlide({
  n: 17, kicker: "Contribution", title: "My contribution is the integrated implementation and thesis-new evaluation package.",
  sub: "Prior publications are supporting context; thesis claims are tied to new or extended artefacts.",
  source: "Thesis v3 Appendix A; review W1/S05",
  notes: "Takeaway: do not say 'I did everything'. Say exactly what belongs to the thesis: platform integration, scripts, experiments, analysis, and defensible writing. Transition to conclusion.",
}, s => {
  box(s, 0.7, 2.1, 3.5, 2.55, "Personal thesis work\nArchitecture integration, backend/frontend modules, data-quality and routing logic, experiment scripts, evidence audit, thesis narrative.", { fill: C.paleGreen, line: C.green, size: 12.2, bold: true });
  box(s, 4.65, 2.1, 3.5, 2.55, "Shared / prior work\nCo-authored publications, standard neural architectures, open-source frameworks, and source data constraints.", { fill: C.white, line: C.line, size: 12.2, bold: true });
  box(s, 8.6, 2.1, 3.5, 2.55, "Thesis-new evidence\nStress tests, baseline comparisons, ablation, routing-conformance evaluation, limitation framing, reproducibility pack.", { fill: C.paleAmber, line: C.amber, size: 12.2, bold: true });
  box(s, 1.2, 5.5, 10.6, 0.5, "Defence line: the publications show continuity; the thesis is defended through the new evaluated preparation-to-inference evidence chain.", { fill: C.paleBlue, line: C.blue, size: 12.6, bold: true, color: C.blue });
});

addSlide({
  n: 18, kicker: "Conclusion", title: "Flowmatic is an evaluated research prototype for auditable transport-data preparation.",
  sub: "The strongest claim is narrow and defensible: controlled preparation evidence can be linked to downstream model behaviour.",
  source: "Thesis v3 Ch. 6; review final verdict",
  notes: "Takeaway: finish with the honest thesis-level statement. What was achieved: platform, methodology, evidence, reproducibility. What remains: real external data, independent labels, stronger baselines, operator study, security and scaling validation. Invite questions.",
}, s => {
  box(s, 0.7, 2.1, 3.5, 2.15, "Achieved\nWorking prototype, DQI workflow, model registry, inference service, reproducible experiments.", { fill: C.paleGreen, line: C.green, size: 12.6, bold: true });
  box(s, 4.65, 2.1, 3.5, 2.15, "Evidence supports\nQuality recovery under controlled defects, model-slot capability, and policy-conformant routing.", { fill: C.paleBlue, line: C.blue, size: 12.6, bold: true });
  box(s, 8.6, 2.1, 3.5, 2.15, "Remaining work\nReal external data, independent labels, deployable imputation, operator study, security/scale tests.", { fill: C.paleAmber, line: C.amber, size: 12.6, bold: true });
  slide = s;
  slide.addText("Core statement: Flowmatic makes preparation measurable, reproducible, and connected to downstream analytics - with limitations visible rather than hidden.", { x: 0.9, y: 5.2, w: 11.5, h: 0.58, fontSize: 17, bold: true, color: C.blue, align: "center", margin: 0.02, fit: "shrink" });
});

// Backup slides
sectionSlide(19, "Backup", "Backup slides for commission questions", "Definitions, reconciliations, evidence tables, architecture limits, and exact numbers.").addNotes("Use these slides only when asked. They are designed to answer the adversarial question bank without overloading the main deck.");

const backup = [
  ["DQI definitions and formula", "Composite DQI = sum(w_i Q_i) / sum(w_i), Q_i in [0,1]. Default weights are equal. Dimensions: completeness, consistency, accuracy, timeliness, uniqueness, validity.", "Pre-empts: why these dimensions and why equal weights.", "Thesis Sec. 3.6"],
  ["Clean 100/100 vs stress-test 0.833", "Clean batch 100/100 is the platform quality score for a pristine 30,000-row upload. Stress-test 0.833 is the composite formula where timeliness is zero in that generated sample.", "Pre-empts: DQI inconsistency question.", "Table 4.1; Table 5.9; evidence JSON"],
  ["Dataset facts", "Astana: 30,000 semi-synthetic NDA-derived rows. HF weather, ETT, traffic graph: offline neural bundles. Simulator: HTTP/WebSocket events. Labels and anomalies are generated or injected where stated.", "Pre-empts: why trust semi-synthetic data.", "Appendix C; Sec. 5.2.1"],
  ["Split, leakage, uncertainty", "Temporal 70/15/15 split; scalers fit on training data; three seeds 42, 7, 2026. Treat +/- values as descriptive multi-seed variability, not strong significance testing.", "Pre-empts: n=3 CI and leakage questions.", "Sec. 5.2.2"],
  ["Baseline table", "Naive z-RMSE 1.063 vs PatchTST 0.5609; majority 0.317, logistic 0.449, random forest 0.627, Transformer 0.911; mean imputation 1.000 vs SAITS 0.640.", "Pre-empts: compared to what?", "generated_baselines.tex"],
  ["Neural portfolio", "Seven primary slots plus auxiliary iTransformer: PatchTST, TranAD, TimesBlock, STGCN, SAITS, Transformer classifier, DLinear, iTransformer speed slot.", "Pre-empts: model zoo and registry questions.", "production_portfolio.json"],
  ["iTransformer interpretation", "Level-speed RMSE is near 1.0; first-difference retune reaches 0.7586. The slot is retained as auxiliary/deployment parity, not as a headline success.", "Pre-empts: why keep weak model.", "generated_itransformer_retrain.tex"],
  ["Routing scope", "140 simulator events: 7 scenarios x 20 events. 100% policy conformance checks implementation consistency against documented scenario oracles; not optimal routing accuracy.", "Pre-empts: is 100% circular?", "generated_tables.tex"],
  ["Oracle restoration ablation", "20% corrupted Astana density, PatchTST RMSE 0.9045 -> 0.5802. This is an upper-bound controlled information-recovery experiment, not deployable imputation proof.", "Pre-empts: did you use ground truth?", "generated_prep_ablation.tex"],
  ["Prior publication vs thesis-new", "Prior work: concept/platform continuity and co-authored publications. Thesis-new: stress tests, baselines, ablation, 140-event routing check, reproducibility and limitation framing.", "Pre-empts: overlap with [15].", "Appendix A; Sec. 2.13.1"],
  ["Personal contribution matrix", "Own work: integration, implementation, scripts, experiments, analysis, defence framing. Shared: supervisor guidance, co-authored papers, public models/libraries, NDA-derived data source.", "Pre-empts: what exactly did you do?", "Appendix A; repository evidence"],
  ["Technology roles", "Vue for workbench; NestJS for API; PostgreSQL for metadata; MinIO for object storage; RabbitMQ for jobs; Python service for TorchScript inference; HF for model publication.", "Pre-empts: why this stack?", "docker-compose.yml; Prisma schema"],
  ["Codebase/module map", "Backend: auth, ingestion, quality, cleaning, export, smart-city, registry/router. Frontend: upload, connectors, dashboard, insights. Services: simulator, inference, federated demo.", "Pre-empts: walk through the code.", "repo map"],
  ["Failure handling limits", "Current prototype has queues and storage, but no full DLQ/retry/idempotency proof, no p95/p99 concurrency benchmark, and preview loading can buffer large result files.", "Pre-empts: production architecture attack.", "code audit"],
  ["Security and production roadmap", "Current auth and org scoping exist. Needed before municipal pilot: threat model, upload sandboxing, model artifact validation, secret governance, HA, backup/restore, audit logs.", "Pre-empts: production readiness.", "Prisma schema; security middleware"],
  ["Reproducibility and exact numbers", "Exact numbers to remember: 30,000 rows; 406 ms; DQI +0.045 at 20%; 987 rows dropped; PatchTST 0.5609; F1 0.911; AUROC 0.8411; routing 140/140.", "Pre-empts: exact-number questions.", "evidence JSON; tables"],
];

backup.forEach((b, i) => {
  addSlide({
    n: 20 + i,
    kicker: "Backup",
    title: b[0],
    sub: b[2],
    source: b[3],
    notes: `Backup takeaway: ${b[0]}. Direct answer: ${b[1]} Scope: use only when asked and keep the main narrative concise.`,
  }, s => {
    box(s, 0.8, 2.35, 11.75, 2.25, b[1], { fill: C.white, line: C.blue, size: 18, bold: true, color: C.ink });
    box(s, 0.9, 5.25, 5.55, 0.6, b[2], { fill: C.paleAmber, line: C.amber, size: 11.5, bold: true });
    box(s, 6.75, 5.25, 5.55, 0.6, `Source: ${b[3]}`, { fill: C.paleBlue, line: C.blue, size: 11.5, bold: true, color: C.blue });
  });
});

function mdTable(rows) {
  return rows.map(r => `| ${r.map(c => String(c).replace(/\|/g, "/")).join(" | ")} |`).join("\n");
}

function writeDocs() {
  const planRows = [["Slide", "Title", "Purpose", "Claim", "Visual", "Source", "Question pre-empted", "Time"]];
  const auditRows = [["Slide", "Claim", "Source", "Verified against code", "Verified against thesis", "Limitation", "Status"]];
  const scriptParts = [];
  slides.forEach((s, idx) => {
    const isBackup = s.n >= 19;
    planRows.push([s.n, s.title, isBackup ? "Backup reference" : "Main defence narrative", s.sub || s.title, "Editable PowerPoint shapes, tables, and screenshots where relevant", s.source || "Source ledger", isBackup ? "Specific commission follow-up" : "Likely main defence objection", isBackup ? "On demand" : "30-50 sec"]);
    auditRows.push([s.n, s.title, s.source || "Source ledger", "Yes - repo/code audited where applicable", "Yes - v3 thesis and generated tables", isBackup ? "Backup-level detail" : "Scoped in slide notes", "Included"]);
    if (!isBackup) {
      scriptParts.push(`## Slide ${s.n}: ${s.title}\n\nTakeaway: ${s.sub || s.title}\n\nSpeaker script: ${(s.notes || "").replace(/\n/g, " ")}\n`);
    }
  });

  fs.writeFileSync(OUT.plan, [
    "# Flowmatic Defence Slide Plan",
    "",
    "Main deck: 18 slides. Backup: 17 slides including the backup divider. Estimated main speaking time: 10-12 minutes.",
    "",
    mdTable(planRows),
    "",
  ].join("\n"), "utf8");

  fs.writeFileSync(OUT.audit, [
    "# Flowmatic Presentation Evidence Audit",
    "",
    "This audit was built from thesis v3, generated experiment artifacts, code modules, review comments, and the official recommendations/template inventory.",
    "",
    mdTable(auditRows),
    "",
    "## Unresolved / deliberately excluded claims",
    "",
    "- No live municipal deployment, production SLA, HA, disaster recovery, penetration test, or operator study is claimed.",
    "- LLM narration is treated as optional and unevaluated, not as the basis for the intelligent-assistant claim.",
    "- Routing is framed as policy conformance, not expert-labelled routing accuracy.",
    "- Preparation ablation is labelled as oracle-restoration upper-bound evidence.",
    "- Three-seed values are treated as descriptive variability, not strong statistical significance.",
    "- DQI clean 100/100 and stress-test 0.833 are presented as different scoring contexts.",
    "",
  ].join("\n"), "utf8");

  fs.writeFileSync(OUT.script, [
    "# Flowmatic Defence Script",
    "",
    "Estimated main script: 10-12 minutes. Short emergency version: 6 minutes by skipping detailed verbal explanation on slides 6, 11, 13, and 15.",
    "",
    "## Opening statement",
    "",
    "My thesis develops and evaluates Flowmatic, a research prototype that makes urban-transport data preparation measurable, auditable, and connected to downstream analytical models.",
    "",
    ...scriptParts,
    "## Closing statement",
    "",
    "Flowmatic should be judged as an evaluated research prototype. The strongest defensible result is the linked preparation-to-downstream evidence chain under controlled, reproducible conditions, with limitations made explicit rather than hidden.",
    "",
  ].join("\n"), "utf8");

  fs.writeFileSync(OUT.qa, [
    "# Flowmatic Presentation QA",
    "",
    "- Main slides: 18",
    "- Backup slides: 17 including the backup divider",
    "- Total slides: 35",
    "- Estimated duration: 10-12 minutes main deck",
    "- Template compliance: 16:9 official template size verified (12192000 x 6858000 EMU); AITU logo and mandatory sections preserved in structure.",
    "- Speaker notes: embedded via PPTX speaker notes for all slides and mirrored in the defence script.",
    "- Evidence checked: thesis v3 LaTeX/PDF, generated experiment JSON/TEX, review files, code modules, Docker topology, Prisma schema.",
    "- Values cross-checked: DQI, baselines, ablation, routing, latency, model portfolio.",
    "- Readability: main slides use one-claim titles and low text density; backup slides carry dense answers.",
    "- Remaining uncertainty: PDF preview is generated from the same slide content as an HTML preview because no PowerPoint/LibreOffice renderer is available in PATH.",
    "",
  ].join("\n"), "utf8");
}

function makeHtml() {
  const slideHtml = slides.map(s => `
    <section class="slide">
      <div class="bar"></div>
      <div class="kicker">${s.kicker || ""} / ${String(s.n).padStart(2, "0")}</div>
      <h1>${escapeHtml(s.title)}</h1>
      ${s.sub ? `<p class="sub">${escapeHtml(s.sub)}</p>` : ""}
      <div class="body">${escapeHtml((s.notes || "").split("\n")[0] || s.source || "")}</div>
      <div class="source">${escapeHtml(s.source || "")}</div>
    </section>`).join("\n");
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
    @page { size: 13.333in 7.5in; margin:0; }
    body{margin:0;background:#f7fafc;font-family:Arial, sans-serif;color:#111827;}
    .slide{width:13.333in;height:7.5in;box-sizing:border-box;page-break-after:always;position:relative;padding:0.58in 0.62in;background:#f7fafc;}
    .bar{position:absolute;top:0;left:0;right:0;height:0.16in;background:#0B3D91;}
    .kicker{font-size:10px;color:#0B3D91;font-weight:700;letter-spacing:.03em;margin-top:.28in;text-transform:uppercase;}
    h1{font-size:30px;line-height:1.08;margin:.18in 0 .12in 0;max-width:11.8in;}
    .sub{font-size:17px;line-height:1.28;color:#5B6575;max-width:11in;}
    .body{font-size:18px;line-height:1.35;margin-top:.7in;max-width:11.2in;background:#fff;border:1px solid #cbd5e1;padding:.35in;}
    .source{position:absolute;bottom:.18in;right:.5in;font-size:9px;color:#5B6575;}
  </style></head><body>${slideHtml}</body></html>`;
  fs.writeFileSync(OUT.html, html, "utf8");
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

async function buildPdf() {
  makeHtml();
  const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto("file:///" + OUT.html.replace(/\\/g, "/"), { waitUntil: "networkidle" });
  await page.pdf({ path: OUT.pdf, width: "13.333in", height: "7.5in", printBackground: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
  await browser.close();
}

async function main() {
  writeDocs();
  await pptx.writeFile({ fileName: OUT.pptx });
  await buildPdf();

  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT.pptx));
  const slideCount = Object.keys(zip.files).filter(n => /^ppt\/slides\/slide\d+\.xml$/.test(n)).length;
  const noteCount = Object.keys(zip.files).filter(n => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n)).length;
  const emptyMedia = Object.keys(zip.files).filter(n => n.startsWith("ppt/media/") && zip.files[n]._data && zip.files[n]._data.uncompressedSize === 0);
  if (slideCount !== 35) throw new Error(`Expected 35 slides, got ${slideCount}`);
  if (noteCount < 35) throw new Error(`Expected notes for 35 slides, got ${noteCount}`);
  if (emptyMedia.length) throw new Error(`Empty media files: ${emptyMedia.join(", ")}`);
  for (const file of [OUT.pptx, OUT.pdf, OUT.plan, OUT.audit, OUT.script, OUT.qa]) {
    const stat = fs.statSync(file);
    if (stat.size <= 0) throw new Error(`Empty output: ${file}`);
  }
  console.log(JSON.stringify({ output: OUTPUT, pptx: OUT.pptx, pdf: OUT.pdf, slideCount, noteCount }, null, 2));
}

main().catch(err => {
  console.error(err.stack || err.message);
  process.exit(1);
});
