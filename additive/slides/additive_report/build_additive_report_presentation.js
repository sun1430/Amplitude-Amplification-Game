"use strict";

const fs = require("node:fs");
const path = require("node:path");
const PptxGenJS = require("pptxgenjs");
const { autoFontSize, calcTextBox } = require("./pptxgenjs_helpers/text");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "University of Waterloo";
pptx.subject = "Additive public opinion shaping report";
pptx.title =
  "Additive Public Opinion Shaping: Exact Quantum Encoding and Observable Estimation";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Barlow Condensed",
  bodyFontFace: "Georgia",
  lang: "en-US",
};

const ROOT = __dirname;
const ADDITIVE_ROOT = path.resolve(ROOT, "..", "..");
const OUT = path.join(ROOT, "additive_report_presentation.pptx");
const SHAPE = pptx.ShapeType;
const CHART = pptx.ChartType;

const COLORS = {
  black: "000000",
  white: "FFFFFF",
  goldLight: "F2EDA8",
  goldBright: "FAE100",
  goldMid: "FED34C",
  goldDark: "EBAB00",
  crimson: "BB0F33",
  gray: "6F6F6F",
  grayDark: "4D4D4D",
  border: "D7CCA3",
  panelFill: "FFFDF5",
  cardFill: "F9F7EF",
  softYellow: "FFF8DE",
  accentFill: "FFF4C9",
  teal: "2A9D8F",
  orange: "E76F51",
};

const FONTS = {
  head: "Franklin Gothic Medium Cond",
  body: "Cambria",
  sans: "Arial",
};

const ASSETS = {
  logoWordmark: path.join(ROOT, "assets", "uw_wordmark.png"),
  background: path.join(ROOT, "assets", "public_opinion_background.png"),
  formulaStateUpdate: path.join(ROOT, "assets", "formula_state_update.png"),
  formulaPayoff: path.join(ROOT, "assets", "formula_payoff.png"),
  formulaMapping: path.join(ROOT, "assets", "formula_mapping.png"),
};

const DATA_FILES = {
  strategy: path.join(ADDITIVE_ROOT, "results", "report", "strategy", "summary.csv"),
  regret: path.join(ADDITIVE_ROOT, "results", "report", "regret", "summary.csv"),
  epsilon: path.join(ADDITIVE_ROOT, "results", "report", "epsilon", "summary.csv"),
  observable: path.join(ADDITIVE_ROOT, "results", "report", "observable_estimation", "aggregate.csv"),
};

const SCENARIO_ORDER = [
  "shallow_low_conflict",
  "medium_conflict",
  "deep_high_conflict",
];

const SCENARIO_LABELS = {
  shallow_low_conflict: "Shallow\nLow Conflict",
  medium_conflict: "Medium\nConflict",
  deep_high_conflict: "Deep\nHigh Conflict",
};

const MODEL_LABELS = {
  ground_truth: "Ground Truth",
  quantum_encoded: "Quantum Encoded",
  residual_mlp: "Residual MLP",
};

const MODEL_COLORS = {
  ground_truth: COLORS.black,
  quantum_encoded: COLORS.goldDark,
  residual_mlp: COLORS.crimson,
};

function parseCsv(csvPath) {
  const raw = fs.readFileSync(csvPath, "utf8").trim();
  const lines = raw.split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx];
    });
    return row;
  });
}

function loadResults() {
  const strategyRows = parseCsv(DATA_FILES.strategy);
  const regretRows = parseCsv(DATA_FILES.regret);
  const epsilonRows = parseCsv(DATA_FILES.epsilon);
  const observableRows = parseCsv(DATA_FILES.observable);

  const strategyByModel = {};
  strategyRows.forEach((row) => {
    strategyByModel[row.model_name] ??= [];
    strategyByModel[row.model_name].push(Number(row.accuracy));
  });

  const regretByModel = {};
  regretRows.forEach((row) => {
    regretByModel[row.model_name] ??= [];
    regretByModel[row.model_name].push(Number(row.mean_regret));
  });

  const epsilonByModel = {};
  epsilonRows.forEach((row) => {
    epsilonByModel[row.model_name] ??= [];
    epsilonByModel[row.model_name].push(Number(row.mean_epsilon));
  });

  const observableByMethod = {};
  observableRows.forEach((row) => {
    observableByMethod[row.method] ??= { labels: [], utility: [] };
    observableByMethod[row.method].labels.push(String(row.query_budget));
    observableByMethod[row.method].utility.push(Number(row.utility_error));
  });

  return {
    strategyByModel,
    regretByModel,
    epsilonByModel,
    observableByMethod,
  };
}

const RESULTS = loadResults();

function addTopBands(slide) {
  slide.background = { color: COLORS.white };
  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.2,
    line: { color: COLORS.black, transparency: 100 },
    fill: { color: COLORS.black },
  });
  const widths = [3.35, 3.35, 3.35, 3.283];
  const fills = [
    COLORS.goldLight,
    COLORS.goldBright,
    COLORS.goldMid,
    COLORS.goldDark,
  ];
  let cursor = 0;
  widths.forEach((width, idx) => {
    slide.addShape(SHAPE.rect, {
      x: cursor,
      y: 0.2,
      w: width,
      h: 0.14,
      line: { color: fills[idx], transparency: 100 },
      fill: { color: fills[idx] },
    });
    cursor += width;
  });
}

function addFooter(slide, pageNo) {
  slide.addText("Additive Public Opinion Shaping", {
    x: 0.38,
    y: 6.99,
    w: 4.25,
    h: 0.18,
    fontFace: FONTS.sans,
    fontSize: 9.5,
    color: COLORS.black,
    margin: 0,
  });
  slide.addText(`PAGE  ${pageNo}`, {
    x: 6.1,
    y: 6.99,
    w: 1.1,
    h: 0.18,
    fontFace: FONTS.sans,
    fontSize: 9.5,
    color: COLORS.black,
    margin: 0,
    align: "center",
  });
  slide.addImage({
    path: ASSETS.logoWordmark,
    ...imageSizingContain(ASSETS.logoWordmark, 10.8, 6.67, 2.1, 0.58),
  });
}

function addTitleChrome(slide) {
  addTopBands(slide);
  slide.addImage({
    path: ASSETS.logoWordmark,
    ...imageSizingContain(ASSETS.logoWordmark, 0.42, 6.15, 3.2, 0.62),
  });
}

function addContentChrome(slide, title, pageNo, subtitle = "") {
  addTopBands(slide);
  slide.addText(title, {
    x: 0.38,
    y: 0.42,
    w: 11.9,
    h: 0.42,
    fontFace: FONTS.head,
    fontSize: 28,
    bold: true,
    color: COLORS.black,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.42,
      y: 0.9,
      w: 9.2,
      h: 0.2,
      fontFace: FONTS.body,
      fontSize: 13.2,
      italic: true,
      color: COLORS.grayDark,
      margin: 0,
    });
  }
  addFooter(slide, pageNo);
}

function addBulletRows(slide, bullets, cfg) {
  let cursorY = cfg.y;
  const fontSize = cfg.fontSize || 20;
  bullets.forEach((bullet) => {
    const metrics = calcTextBox(fontSize, {
      text: bullet,
      w: cfg.w - 0.25,
      fontFace: cfg.fontFace || FONTS.body,
      leading: cfg.leading || 1.12,
      margin: 0,
      padding: 0.02,
    });
    const rowH = Math.max(0.34, metrics.h + 0.04);
    slide.addShape(SHAPE.rect, {
      x: cfg.x,
      y: cursorY + 0.1,
      w: 0.08,
      h: 0.08,
      line: {
        color: cfg.bulletColor || COLORS.goldDark,
        transparency: 100,
      },
      fill: { color: cfg.bulletColor || COLORS.goldDark },
    });
    slide.addText(bullet, {
      x: cfg.x + 0.16,
      y: cursorY,
      w: cfg.w - 0.16,
      h: rowH,
      fontFace: cfg.fontFace || FONTS.body,
      fontSize,
      color: cfg.color || COLORS.black,
      margin: 0,
      leading: cfg.leading || 1.12,
      valign: "top",
      bold: cfg.bold || false,
    });
    cursorY += rowH + (cfg.gap || 0.12);
  });
}

function addCard(slide, cfg) {
  slide.addShape(SHAPE.roundRect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    rectRadius: 0.04,
    line: { color: cfg.lineColor || COLORS.border, pt: 1.1 },
    fill: { color: cfg.fill || COLORS.panelFill },
  });
  slide.addShape(SHAPE.rect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: 0.08,
    line: { color: cfg.accent || COLORS.goldDark, transparency: 100 },
    fill: { color: cfg.accent || COLORS.goldDark },
  });
  slide.addText(cfg.title, {
    x: cfg.x + 0.15,
    y: cfg.y + 0.16,
    w: cfg.w - 0.3,
    h: 0.24,
    fontFace: FONTS.head,
    fontSize: cfg.titleSize || 18,
    bold: true,
    color: COLORS.black,
    margin: 0,
  });
  const bodyOpts = autoFontSize(cfg.body, FONTS.body, {
    x: cfg.x + 0.15,
    y: cfg.y + 0.58,
    w: cfg.w - 0.3,
    h: cfg.h - 0.74,
    fontSize: cfg.bodySize || 15,
    minFontSize: cfg.minBodySize || 12,
    maxFontSize: cfg.bodySize || 15,
    mode: "shrink",
    margin: 0,
    leading: 1.12,
  });
  slide.addText(cfg.body, {
    ...bodyOpts,
    fontFace: cfg.bodyFontFace || FONTS.body,
    color: cfg.bodyColor || COLORS.grayDark,
    margin: 0,
    valign: "top",
  });
}

function addCallout(slide, cfg) {
  slide.addShape(SHAPE.roundRect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    rectRadius: 0.04,
    line: { color: cfg.lineColor || COLORS.crimson, pt: 1.2 },
    fill: { color: cfg.fill || COLORS.softYellow },
  });
  const textOpts = autoFontSize(cfg.text, FONTS.body, {
    x: cfg.x + 0.16,
    y: cfg.y + 0.12,
    w: cfg.w - 0.32,
    h: cfg.h - 0.24,
    fontSize: cfg.fontSize || 15.5,
    minFontSize: cfg.minFontSize || 12.5,
    maxFontSize: cfg.fontSize || 15.5,
    mode: "shrink",
    margin: 0,
    leading: 1.1,
    bold: cfg.bold || false,
    italic: cfg.italic || false,
  });
  slide.addText(cfg.text, {
    ...textOpts,
    fontFace: FONTS.body,
    color: cfg.color || COLORS.black,
    margin: 0,
    bold: cfg.bold || false,
    italic: cfg.italic || false,
  });
}

function addEquationCard(slide, cfg) {
  slide.addShape(SHAPE.roundRect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    rectRadius: 0.04,
    line: { color: cfg.lineColor || COLORS.border, pt: 1.1 },
    fill: { color: cfg.fill || COLORS.panelFill },
  });
  slide.addShape(SHAPE.rect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: 0.08,
    line: { color: cfg.accent || COLORS.goldDark, transparency: 100 },
    fill: { color: cfg.accent || COLORS.goldDark },
  });
  slide.addText(cfg.title, {
    x: cfg.x + 0.16,
    y: cfg.y + 0.14,
    w: cfg.w - 0.32,
    h: 0.28,
    fontFace: FONTS.head,
    fontSize: cfg.titleSize || 17,
    bold: true,
    color: COLORS.black,
    margin: 0,
  });
  if (cfg.formulaPath) {
    slide.addImage({
      path: cfg.formulaPath,
      ...imageSizingContain(
        cfg.formulaPath,
        cfg.x + 0.16,
        cfg.y + 0.44,
        cfg.w - 0.32,
        cfg.h - 0.52
      ),
    });
  } else {
    const formulaOpts = autoFontSize(cfg.formulaText, "Cambria Math", {
      x: cfg.x + 0.2,
      y: cfg.y + 0.58,
      w: cfg.w - 0.4,
      h: cfg.h - 0.78,
      fontSize: cfg.formulaSize || 22,
      minFontSize: cfg.minFormulaSize || 18,
      maxFontSize: cfg.formulaSize || 22,
      mode: "shrink",
      margin: 0,
      leading: 1.0,
    });
    slide.addText(cfg.formulaText, {
      ...formulaOpts,
      fontFace: "Cambria Math",
      color: COLORS.black,
      margin: 0,
      align: "center",
      valign: "mid",
    });
  }
}

function addImagePanel(slide, imgPath, cfg) {
  slide.addShape(SHAPE.roundRect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    rectRadius: 0.03,
    line: { color: COLORS.border, pt: 1.0 },
    fill: { color: COLORS.white },
  });
  slide.addImage({
    path: imgPath,
    ...imageSizingContain(
      imgPath,
      cfg.x + 0.08,
      cfg.y + 0.08,
      cfg.w - 0.16,
      cfg.h - 0.16
    ),
  });
}

function addNode(slide, cfg) {
  slide.addShape(SHAPE.roundRect, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    rectRadius: 0.03,
    line: { color: cfg.lineColor || COLORS.border, pt: 1.0 },
    fill: { color: cfg.fill || COLORS.cardFill },
  });
  const opts = autoFontSize(cfg.text, FONTS.sans, {
    x: cfg.x + 0.08,
    y: cfg.y + 0.07,
    w: cfg.w - 0.16,
    h: cfg.h - 0.14,
    fontSize: cfg.fontSize || 14,
    minFontSize: 11,
    maxFontSize: cfg.fontSize || 14,
    mode: "shrink",
    margin: 0,
    bold: cfg.bold || false,
  });
  slide.addText(cfg.text, {
    ...opts,
    fontFace: cfg.fontFace || FONTS.sans,
    color: cfg.color || COLORS.black,
    margin: 0,
    align: "center",
    valign: "mid",
    bold: cfg.bold || false,
  });
}

function addArrow(slide, x1, y1, x2, y2, color = COLORS.black) {
  const x = Math.min(x1, x2);
  const y = Math.min(y1, y2);
  slide.addShape(SHAPE.line, {
    x,
    y,
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
    flipH: x2 < x1,
    flipV: y2 < y1,
    line: {
      color,
      pt: 1.4,
      endArrowType: "triangle",
    },
  });
}

function addClusteredColumnChart(slide, cfg) {
  const data = cfg.series.map((series) => ({
    name: series.name,
    labels: cfg.labels,
    values: series.values,
  }));
  slide.addChart(CHART.bar, data, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    catAxisLabelFontFace: FONTS.sans,
    catAxisLabelFontSize: 16,
    valAxisLabelFontFace: FONTS.sans,
    valAxisLabelFontSize: 14,
    showTitle: false,
    showLegend: true,
    legendFontFace: FONTS.sans,
    legendFontSize: 14,
    legendPos: "b",
    chartColors: cfg.series.map((series) => series.color),
    valAxisTitle: cfg.valAxisTitle || "",
    valAxisMinVal: cfg.minVal,
    valAxisMaxVal: cfg.maxVal,
    valAxisMajorUnit: cfg.majorUnit,
    valGridLine: { color: "D9D9D9", pt: 1 },
    gapWidthPct: 90,
    overlap: 0,
    orientation: "col",
    border: { color: COLORS.border, pt: 1 },
  });
}

function addLineChart(slide, cfg) {
  const data = cfg.series.map((series) => ({
    name: series.name,
    labels: cfg.labels,
    values: series.values,
  }));
  slide.addChart(CHART.line, data, {
    x: cfg.x,
    y: cfg.y,
    w: cfg.w,
    h: cfg.h,
    catAxisLabelFontFace: FONTS.sans,
    catAxisLabelFontSize: 16,
    valAxisLabelFontFace: FONTS.sans,
    valAxisLabelFontSize: 14,
    showTitle: false,
    showLegend: true,
    legendFontFace: FONTS.sans,
    legendFontSize: 14,
    legendPos: "b",
    chartColors: cfg.series.map((series) => series.color),
    valAxisTitle: cfg.valAxisTitle || "",
    catAxisTitle: cfg.catAxisTitle || "",
    valAxisMinVal: cfg.minVal,
    valAxisMaxVal: cfg.maxVal,
    valAxisMajorUnit: cfg.majorUnit,
    valGridLine: { color: "D9D9D9", pt: 1 },
    lineSize: 3,
    markerSize: 7,
    smoothLine: false,
    border: { color: COLORS.border, pt: 1 },
  });
}

function addManualTable(slide, cfg) {
  const rowCount = cfg.rows.length;
  const colCount = cfg.rows[0].length;
  const cellW = cfg.w / colCount;
  const cellH = cfg.h / rowCount;
  for (let r = 0; r < rowCount; r += 1) {
    for (let c = 0; c < colCount; c += 1) {
      const isHeader = r === 0 || c === 0;
      slide.addShape(SHAPE.rect, {
        x: cfg.x + c * cellW,
        y: cfg.y + r * cellH,
        w: cellW,
        h: cellH,
        line: { color: COLORS.border, pt: 1 },
        fill: { color: isHeader ? COLORS.accentFill : COLORS.white },
      });
      slide.addText(String(cfg.rows[r][c]), {
        x: cfg.x + c * cellW + 0.06,
        y: cfg.y + r * cellH + 0.05,
        w: cellW - 0.12,
        h: cellH - 0.1,
        fontFace: isHeader ? FONTS.sans : FONTS.body,
        fontSize: isHeader ? 14 : 13,
        bold: isHeader,
        color: COLORS.black,
        margin: 0,
        align: c === 0 ? "left" : "center",
        valign: "mid",
      });
    }
  }
}

function finalizeSlide(slide) {
  warnIfSlideHasOverlaps(slide, pptx, { muteContainment: true });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function buildTitleSlide() {
  const slide = pptx.addSlide();
  addTitleChrome(slide);

  slide.addText("ADDITIVE PUBLIC OPINION SHAPING", {
    x: 0.6,
    y: 0.9,
    w: 6.0,
    h: 0.6,
    fontFace: FONTS.head,
    fontSize: 30,
    bold: true,
    color: COLORS.black,
    margin: 0,
  });
  slide.addText("Exact Quantum Encoding and Observable Estimation", {
    x: 0.62,
    y: 1.55,
    w: 5.9,
    h: 0.4,
    fontFace: FONTS.body,
    fontSize: 21,
    italic: true,
    color: COLORS.grayDark,
    margin: 0,
  });
  slide.addText(
    "This deck isolates the additive mainline, the exact strategic encoding result, and the observable-estimation benchmark.",
    {
      x: 0.64,
      y: 2.15,
      w: 5.6,
      h: 0.72,
      fontFace: FONTS.body,
      fontSize: 19,
      color: COLORS.black,
      margin: 0,
      leading: 1.12,
    }
  );

  addNode(slide, {
    x: 6.95,
    y: 1.4,
    w: 1.45,
    h: 0.72,
    text: "Actors",
    fill: COLORS.cardFill,
    bold: true,
    fontSize: 15,
  });
  addNode(slide, {
    x: 8.8,
    y: 1.4,
    w: 1.7,
    h: 0.72,
    text: "Mixed Narratives",
    fill: COLORS.accentFill,
    bold: true,
    fontSize: 14,
  });
  addNode(slide, {
    x: 10.95,
    y: 1.4,
    w: 1.5,
    h: 0.72,
    text: "Observable Payoff",
    fill: COLORS.softYellow,
    bold: true,
    fontSize: 14,
  });
  addArrow(slide, 8.46, 1.76, 8.74, 1.76, COLORS.black);
  addArrow(slide, 10.56, 1.76, 10.89, 1.76, COLORS.black);

  addCard(slide, {
    x: 6.88,
    y: 2.55,
    w: 5.55,
    h: 2.05,
    title: "Core message",
    body: "In the additive setting, quantum does not define a different game. It exactly encodes the classical ground truth and gives a cleaner route to observable-payoff estimation.",
    accent: COLORS.crimson,
    bodySize: 16,
    minBodySize: 14,
  });
  addCallout(slide, {
    x: 6.9,
    y: 4.95,
    w: 5.48,
    h: 0.88,
    text: "Technical scope: additive GT, exact quantum encoding, residual MLP surrogate, and AE benchmark.",
    fontSize: 16,
    fill: COLORS.accentFill,
    lineColor: COLORS.goldDark,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildBackgroundSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "PROBLEM BACKGROUND",
    2,
    "Why additive public-opinion shaping is the right reduction."
  );
  addImagePanel(slide, ASSETS.background, {
    x: 7.05,
    y: 1.2,
    w: 5.75,
    h: 4.8,
  });
  addBulletRows(
    slide,
    [
      "Multiple actors intervene in one shared opinion space rather than in isolated channels.",
      "Each coordinate represents a narrative axis such as economic framing, identity, or social value.",
      "Actors allocate influence across those coordinates, so strategy is a distributional decision, not a single move.",
      "Platform redistribution and audience cognition mix narratives across coordinates, which creates strategic coupling.",
    ],
    {
      x: 0.7,
      y: 1.35,
      w: 5.95,
      fontSize: 18.5,
      gap: 0.12,
    }
  );
  addCallout(slide, {
    x: 0.74,
    y: 5.52,
    w: 5.9,
    h: 0.58,
    text: "Additive modeling makes this problem explicit: it turns shared narrative shaping into a state update plus a measurable strategic payoff.",
    fontSize: 15.5,
    fill: COLORS.softYellow,
    lineColor: COLORS.crimson,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildFormulationSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "ADDITIVE MAINLINE FORMULATION",
    3,
    "Three models share one distribution-based scoring interface."
  );
  addCard(slide, {
    x: 0.65,
    y: 1.22,
    w: 3.9,
    h: 1.12,
    title: "ClassicalGroundTruthGame",
    body: "Reference additive dynamics with explicit mixing matrix M and agent-specific maps Bᵢ.",
    accent: COLORS.black,
    bodySize: 16.5,
    minBodySize: 15,
    bodyFontFace: FONTS.sans,
  });
  addCard(slide, {
    x: 0.65,
    y: 2.52,
    w: 3.9,
    h: 1.12,
    title: "QuantumEncodedGame",
    body: "Exact encoding of the same additive game in a quantum-style latent state.",
    accent: COLORS.goldDark,
    bodySize: 16.0,
    minBodySize: 14.5,
    bodyFontFace: FONTS.sans,
  });
  addCard(slide, {
    x: 0.65,
    y: 3.82,
    w: 3.9,
    h: 1.12,
    title: "ResidualMLPSurrogate",
    body: "Learned classical surrogate from joint actions to terminal influence distributions.",
    accent: COLORS.crimson,
    bodySize: 16.0,
    minBodySize: 14.5,
    bodyFontFace: FONTS.sans,
  });
  addEquationCard(slide, {
    x: 5.0,
    y: 1.22,
    w: 7.65,
    h: 1.35,
    title: "Additive ground-truth dynamics",
    formulaPath: ASSETS.formulaStateUpdate,
    accent: COLORS.black,
  });
  addEquationCard(slide, {
    x: 5.0,
    y: 2.76,
    w: 7.65,
    h: 1.35,
    title: "Observable-based payoff",
    formulaPath: ASSETS.formulaPayoff,
    accent: COLORS.goldDark,
  });
  addCallout(slide, {
    x: 5.08,
    y: 4.48,
    w: 7.48,
    h: 0.9,
    text: "entmax-1.5 is sharper than softmax, permits sparse attention patterns, and avoids an overly smooth surrogate target.",
    fontSize: 15.2,
    fill: COLORS.accentFill,
    lineColor: COLORS.goldDark,
  });
  finalizeSlide(slide);
}

function buildStatementSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "RIGOROUS STATEMENT",
    4,
    "What exact quantum encoding does and does not claim."
  );
  addCard(slide, {
    x: 0.72,
    y: 1.22,
    w: 11.9,
    h: 1.55,
    title: "Exact statement",
    body: "For every feasible joint action a, the additive QuantumEncodedGame induces the same terminal influence distribution as the ClassicalGroundTruthGame. Therefore, best responses, regret, and epsilon-equilibrium properties coincide, up to amplitude-estimation discretization error when utilities are evaluated via observable expectations.",
    accent: COLORS.crimson,
    bodySize: 16.5,
    minBodySize: 14,
  });
  addNode(slide, {
    x: 1.1,
    y: 3.3,
    w: 1.55,
    h: 0.82,
    text: "pₜ",
    fill: COLORS.accentFill,
    bold: true,
    fontSize: 18,
  });
  slide.addShape(SHAPE.roundRect, {
    x: 3.15,
    y: 3.3,
    w: 2.35,
    h: 0.82,
    rectRadius: 0.03,
    line: { color: COLORS.border, pt: 1.0 },
    fill: { color: COLORS.cardFill },
  });
  slide.addImage({
    path: ASSETS.formulaMapping,
    ...imageSizingContain(ASSETS.formulaMapping, 3.24, 3.42, 2.17, 0.56),
  });
  addNode(slide, {
    x: 6.0,
    y: 3.3,
    w: 2.45,
    h: 0.82,
    text: "Exact additive update preserved",
    fill: COLORS.softYellow,
    bold: true,
    fontSize: 14,
  });
  addNode(slide, {
    x: 9.0,
    y: 3.3,
    w: 2.7,
    h: 0.82,
    text: "Observable payoff from encoded state",
    fill: COLORS.accentFill,
    bold: true,
    fontSize: 14,
  });
  addArrow(slide, 2.72, 3.71, 3.06, 3.71, COLORS.black);
  addArrow(slide, 5.57, 3.71, 5.92, 3.71, COLORS.black);
  addArrow(slide, 8.5, 3.71, 8.92, 3.71, COLORS.black);
  addCallout(slide, {
    x: 1.05,
    y: 4.65,
    w: 10.65,
    h: 0.82,
    text: "This is an exact strategic encoding result. It shows model equivalence at the distribution level. It does not, by itself, establish end-to-end wall-clock quantum speedup.",
    fontSize: 15.5,
    fill: COLORS.softYellow,
    lineColor: COLORS.crimson,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildSetupSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "EXPERIMENTAL SETUP",
    5,
    "All headline metrics are judged against the additive classical ground truth."
  );
  addCard(slide, {
    x: 0.78,
    y: 1.28,
    w: 3.65,
    h: 1.08,
    title: "Scenarios",
    body: "Three additive scenarios: shallow, medium, and deep conflict.",
    accent: COLORS.black,
    bodySize: 16.5,
    minBodySize: 15,
    bodyFontFace: FONTS.sans,
  });
  addCard(slide, {
    x: 4.84,
    y: 1.28,
    w: 3.65,
    h: 1.08,
    title: "Headline metrics",
    body: "Accuracy, Regret, and Epsilon are all checked against additive GT.",
    accent: COLORS.goldDark,
    bodySize: 16.5,
    minBodySize: 15,
    bodyFontFace: FONTS.sans,
  });
  addCard(slide, {
    x: 8.9,
    y: 1.28,
    w: 3.65,
    h: 1.08,
    title: "Estimation benchmark",
    body: "AE is compared to Monte Carlo on observable-payoff estimation at equal query budgets.",
    accent: COLORS.crimson,
    bodySize: 16.0,
    minBodySize: 15,
    bodyFontFace: FONTS.sans,
  });
  addNode(slide, {
    x: 1.15,
    y: 3.0,
    w: 1.95,
    h: 0.82,
    text: "Ground Truth",
    fill: COLORS.accentFill,
    bold: true,
    fontSize: 16,
  });
  addNode(slide, {
    x: 3.55,
    y: 3.0,
    w: 2.2,
    h: 0.82,
    text: "Quantum Encoded",
    fill: COLORS.cardFill,
    bold: true,
    fontSize: 16,
  });
  addNode(slide, {
    x: 6.2,
    y: 3.0,
    w: 2.0,
    h: 0.82,
    text: "Residual MLP",
    fill: COLORS.softYellow,
    bold: true,
    fontSize: 16,
  });
  addNode(slide, {
    x: 8.65,
    y: 3.0,
    w: 3.0,
    h: 0.82,
    text: "Evaluate under GT reference",
    fill: COLORS.accentFill,
    bold: true,
    fontSize: 16,
  });
  addArrow(slide, 3.18, 3.41, 3.47, 3.41, COLORS.black);
  addArrow(slide, 5.84, 3.41, 6.12, 3.41, COLORS.black);
  addArrow(slide, 8.28, 3.41, 8.57, 3.41, COLORS.black);
  addBulletRows(
    slide,
    [
      "Accuracy: whether the approximate model selects the same best-response candidate as the ground truth.",
      "Regret: exact improvement still available in the ground-truth game after following the candidate recommendation.",
      "Epsilon: epsilon-Nash slack of the candidate equilibrium when it is checked in the ground-truth game.",
    ],
    {
      x: 0.92,
      y: 4.3,
      w: 11.25,
      fontSize: 18,
      gap: 0.12,
    }
  );
  finalizeSlide(slide);
}

function buildStrategyRegretSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "STRATEGY AND REGRET RESULTS",
    6,
    "Native PowerPoint charts rebuilt from additive report summaries."
  );
  addClusteredColumnChart(slide, {
    x: 0.55,
    y: 1.3,
    w: 5.95,
    h: 3.45,
    labels: SCENARIO_ORDER.map((key) => SCENARIO_LABELS[key]),
    series: ["ground_truth", "quantum_encoded", "residual_mlp"].map((key) => ({
      name: MODEL_LABELS[key],
      values: RESULTS.strategyByModel[key],
      color: MODEL_COLORS[key],
    })),
    minVal: 0,
    maxVal: 1.05,
    majorUnit: 0.2,
    valAxisTitle: "Accuracy",
  });
  addClusteredColumnChart(slide, {
    x: 6.82,
    y: 1.3,
    w: 5.95,
    h: 3.45,
    labels: SCENARIO_ORDER.map((key) => SCENARIO_LABELS[key]),
    series: ["ground_truth", "quantum_encoded", "residual_mlp"].map((key) => ({
      name: MODEL_LABELS[key],
      values: RESULTS.regretByModel[key],
      color: MODEL_COLORS[key],
    })),
    minVal: 0,
    maxVal: 0.012,
    majorUnit: 0.002,
    valAxisTitle: "Mean Regret",
  });
  addCard(slide, {
    x: 0.78,
    y: 4.98,
    w: 5.35,
    h: 0.88,
    title: "Accuracy numbers",
    body: "Quantum: 1.000 / 1.000 / 0.984.   MLP: 0.859 / 0.906 / 0.953.",
    accent: COLORS.goldDark,
    bodySize: 15.2,
  });
  addCard(slide, {
    x: 6.98,
    y: 4.98,
    w: 5.35,
    h: 0.88,
    title: "Regret numbers",
    body: "Quantum: 8.9e-06 / 0 / 1.0e-04.   MLP: 1.04e-02 / 2.72e-03 / 3.66e-03.",
    accent: COLORS.crimson,
    bodySize: 14.4,
  });
  addCallout(slide, {
    x: 0.82,
    y: 5.88,
    w: 11.4,
    h: 0.5,
    text: "Exact quantum encoding preserves strategy structure, while the learned surrogate remains approximate.",
    fontSize: 13.5,
    fill: COLORS.softYellow,
    lineColor: COLORS.crimson,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildEpsilonSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "EPSILON RESULT AND INTERPRETATION",
    7,
    "Epsilon is a structural consistency check, not the only ranking criterion."
  );
  addManualTable(slide, {
    x: 0.8,
    y: 1.45,
    w: 7.2,
    h: 2.5,
    rows: [
      ["Scenario", "Ground Truth", "Quantum", "Residual MLP"],
      ["Shallow Low Conflict", "0.0713", "0.0713", "0.0000"],
      ["Medium Conflict", "0.0000", "0.0000", "0.0000"],
      ["Deep High Conflict", "0.0126", "0.0126", "0.0126"],
    ],
  });
  addCard(slide, {
    x: 8.45,
    y: 1.55,
    w: 3.75,
    h: 1.15,
    title: "Reading the table",
    body: "Quantum matches GT because the additive encoding preserves the same terminal distribution.",
    accent: COLORS.goldDark,
    bodySize: 15.5,
    minBodySize: 15,
    bodyFontFace: FONTS.sans,
  });
  addCard(slide, {
    x: 8.45,
    y: 2.95,
    w: 3.75,
    h: 1.15,
    title: "MLP caveat",
    body: "MLP is learned, not exact. A low epsilon in one scenario does not certify equivalence.",
    accent: COLORS.crimson,
    bodySize: 15.5,
    minBodySize: 14.5,
    bodyFontFace: FONTS.sans,
  });
  addCallout(slide, {
    x: 0.84,
    y: 4.45,
    w: 11.35,
    h: 0.78,
    text: "Epsilon is reported here to check structural consistency. It should be read together with Accuracy and Regret rather than as a standalone ranking metric.",
    fontSize: 16,
    fill: COLORS.accentFill,
    lineColor: COLORS.goldDark,
    bold: true,
  });
  addBulletRows(
    slide,
    [
      "Quantum and ground truth coincide because they induce the same terminal influence distribution in the additive construction.",
      "Residual MLP remains useful as a modeling baseline, but it should be interpreted as a learned surrogate rather than as a solver or an exact reformulation.",
    ],
    {
      x: 0.98,
      y: 5.3,
      w: 11.1,
      fontSize: 17.2,
      gap: 0.08,
    }
  );
  finalizeSlide(slide);
}

function buildObservableSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "OBSERVABLE ESTIMATION BENCHMARK",
    8,
    "Amplitude Estimation vs Monte Carlo under matched query budgets."
  );
  addLineChart(slide, {
    x: 0.62,
    y: 1.32,
    w: 7.35,
    h: 4.15,
    labels: RESULTS.observableByMethod.amplitude_estimation.labels,
    series: [
      {
        name: "AE utility error",
        values: RESULTS.observableByMethod.amplitude_estimation.utility,
        color: COLORS.orange,
      },
      {
        name: "MC utility error",
        values: RESULTS.observableByMethod.monte_carlo.utility,
        color: COLORS.teal,
      },
    ],
    minVal: 0,
    maxVal: 0.03,
    majorUnit: 0.005,
    valAxisTitle: "Utility Error",
    catAxisTitle: "Query Budget",
  });
  addCard(slide, {
    x: 8.35,
    y: 1.52,
    w: 3.85,
    h: 0.96,
    title: "Budget 8",
    body: "AE 0.0190 vs MC 0.0266",
    accent: COLORS.orange,
    bodySize: 15,
  });
  addCard(slide, {
    x: 8.35,
    y: 2.72,
    w: 3.85,
    h: 0.96,
    title: "Budget 64",
    body: "AE 0.00255 vs MC 0.00919",
    accent: COLORS.orange,
    bodySize: 15,
  });
  addCard(slide, {
    x: 8.35,
    y: 3.92,
    w: 3.85,
    h: 0.96,
    title: "Budget 256",
    body: "AE 0.000579 vs MC 0.00475",
    accent: COLORS.orange,
    bodySize: 15,
  });
  addCallout(slide, {
    x: 0.88,
    y: 5.82,
    w: 11.25,
    h: 0.42,
    text: "For this observable-estimation task, amplitude estimation achieves lower estimation error than classical sampling at every tested query budget.",
    fontSize: 14.5,
    fill: COLORS.softYellow,
    lineColor: COLORS.crimson,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildSupportSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "WHAT THE RESULTS SUPPORT",
    9,
    "A strict claim boundary is part of the contribution."
  );
  addCard(slide, {
    x: 0.82,
    y: 1.35,
    w: 5.55,
    h: 3.85,
    title: "Supported",
    body: "1. Exact quantum encoding preserves additive strategic structure.\n\n2. Best responses, regret, and epsilon agree with GT up to AE discretization.\n\n3. AE beats Monte Carlo on observable-payoff error at equal query budgets.",
    accent: COLORS.goldDark,
    bodySize: 16.5,
    minBodySize: 15,
  });
  addCard(slide, {
    x: 6.92,
    y: 1.35,
    w: 5.55,
    h: 3.85,
    title: "Not Supported",
    body: "1. Universal quantum dominance over classical methods.\n\n2. Hardware-level wall-clock speedup claims.\n\n3. Claims beyond the additive estimation benchmark.",
    accent: COLORS.crimson,
    bodySize: 16.5,
    minBodySize: 15,
  });
  addCallout(slide, {
    x: 1.05,
    y: 5.58,
    w: 11.0,
    h: 0.55,
    text: "This boundary makes the additive story defensible: exact encoding is a structural statement, while AE provides the measurable performance advantage on a specific observable-estimation task.",
    fontSize: 15,
    fill: COLORS.accentFill,
    lineColor: COLORS.goldDark,
    bold: true,
  });
  finalizeSlide(slide);
}

function buildConclusionSlide() {
  const slide = pptx.addSlide();
  addContentChrome(
    slide,
    "CONCLUSION",
    10,
    "Final takeaways for the additive-only report."
  );
  addCard(slide, {
    x: 0.7,
    y: 1.35,
    w: 3.8,
    h: 1.25,
    title: "Takeaway 1",
    body: "Additive ground truth makes public-opinion shaping explicit and strategically testable.",
    accent: COLORS.black,
    bodySize: 16,
  });
  addCard(slide, {
    x: 4.78,
    y: 1.35,
    w: 3.8,
    h: 1.25,
    title: "Takeaway 2",
    body: "Quantum encoding is exact at the strategic level in this additive construction.",
    accent: COLORS.goldDark,
    bodySize: 16,
  });
  addCard(slide, {
    x: 8.85,
    y: 1.35,
    w: 3.8,
    h: 1.25,
    title: "Takeaway 3",
    body: "Amplitude estimation gives the clean performance advantage in measurable observable-payoff estimation.",
    accent: COLORS.crimson,
    bodySize: 15.5,
  });
  addCard(slide, {
    x: 0.92,
    y: 3.15,
    w: 11.2,
    h: 1.4,
    title: "Oral closing statement",
    body: "In the additive setting, the quantum contribution is an exact strategic encoding plus a cleaner route to observable-payoff estimation.",
    accent: COLORS.goldDark,
    bodySize: 18,
    minBodySize: 16,
  });
  addBulletRows(
    slide,
    [
      "This additive deck is intentionally narrower than the full route-2 report and avoids legacy-baseline claims in its main conclusion.",
      "The rigorous claim is about preserved strategic structure and benchmarked observable estimation, not about universal quantum advantage.",
    ],
    {
      x: 1.05,
      y: 5.0,
      w: 10.95,
      fontSize: 18,
      gap: 0.12,
    }
  );
  finalizeSlide(slide);
}

function buildSlides() {
  buildTitleSlide();
  buildBackgroundSlide();
  buildFormulationSlide();
  buildStatementSlide();
  buildSetupSlide();
  buildStrategyRegretSlide();
  buildEpsilonSlide();
  buildObservableSlide();
  buildSupportSlide();
  buildConclusionSlide();
}

async function main() {
  buildSlides();
  await pptx.writeFile({ fileName: OUT, compression: false });
  console.log(`Wrote ${OUT}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
