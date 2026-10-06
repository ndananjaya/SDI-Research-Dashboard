import fs from "node:fs";
import path from "node:path";
import XLSX from "xlsx";

const root = path.resolve(import.meta.dirname, "..");
const bundledDataRoot = path.join(root, "datasets");
const legacyDataRoot = path.resolve(root, "..", "SDI_Assignment", "datasets");
// Vercel builds only the dashboard project, so the portable copy is preferred.
// The sibling path keeps existing local checkouts working during the transition.
const dataRoot = fs.existsSync(bundledDataRoot) ? bundledDataRoot : legacyDataRoot;
const rawFile = process.env.RAW_DATASET ?? path.join(dataRoot, "uncleaned_response_dataset.xlsx");
const cleanFile = process.env.CLEAN_DATASET ?? path.join(dataRoot, "cleaned_response_dataset.xlsx");
const outputFile = path.join(root, "public", "data", "dashboard.json");

const normalise = (value) => String(value ?? "").toLowerCase().replace(/\s+/g, " ").trim();
const aliases = {
  timestamp: ["timestamp"], faculty: ["faculty"], device_type: ["device_type", "preferred device type"], brand_name: ["brand_name", "brand name"],
  chosen_method: ["chosen_method", "how did you choose"], battery_life: ["normal_use_battery_life", "battery life time"], budget: ["budget", "allocated budget"],
  device_price: ["device_price", "actual price"], payment_method: ["payment_method", "payment method"], storage: ["storage_capacity", "total storage capacity"],
  lifespan: ["expected_life_span", "expected device lifespan"], research_time: ["research_time", "research time"], primary_influence: ["primary_influence", "primary factor influencing"],
  weight: ["weight(g)", "weight of the device"], store_type: ["store_type", "where did you purchase"], satisfaction: ["satisfaction", "are you satisfied"], ram: ["ram_capacity", "ram size"],
};
const readWorkbook = (file) => {
  const book = XLSX.readFile(file, { cellDates: true });
  const sheet = book.Sheets[book.SheetNames[0]];
  const rows = XLSX.utils.sheet_to_json(sheet, { defval: null, raw: true });
  return rows.map((row) => {
    const out = {};
    for (const [key, value] of Object.entries(row)) {
      if (normalise(key).startsWith("unnamed") || normalise(key) === "__empty") continue;
      const found = Object.entries(aliases).find(([, terms]) => terms.some((term) => normalise(key).includes(term)));
      out[found?.[0] ?? key] = value;
    }
    return out;
  });
};
const sourceColumnCount = (file) => {
  const book = XLSX.readFile(file);
  const sheet = book.Sheets[book.SheetNames[0]];
  const headers = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: null })[0] || [];
  return headers.filter((header) => header != null && normalise(header) !== "__empty").length;
};
const numeric = (value) => { const n = Number(value); return Number.isFinite(n) ? n : null; };
const round = (value, digits = 4) => value == null || !Number.isFinite(value) ? null : Number(value.toFixed(digits));
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const median = (values) => { const sorted = [...values].sort((a, b) => a - b); const middle = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2; };
const std = (values) => values.length < 2 ? 0 : Math.sqrt(values.reduce((sum, value) => sum + (value - mean(values)) ** 2, 0) / (values.length - 1));
const min = (values) => Math.min(...values); const max = (values) => Math.max(...values);
const quantile = (values, q) => { const sorted = [...values].sort((a, b) => a - b); const index = (sorted.length - 1) * q; const lo = Math.floor(index); const hi = Math.ceil(index); return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo); };
const valueCounts = (rows, key) => { const map = new Map(); rows.forEach((row) => { const label = row[key] == null ? "Not recorded" : String(row[key]); map.set(label, (map.get(label) ?? 0) + 1); }); return [...map].map(([label, count]) => ({ label, count, percentage: round(count / rows.length * 100, 2) })).sort((a, b) => b.count - a.count); };
const normalCdf = (x) => { const t = 1 / (1 + .2316419 * Math.abs(x)); const d = .3989423 * Math.exp(-x * x / 2); const p = 1 - d * t * (.3193815 + t * (-.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return x >= 0 ? p : 1 - p; };
const rmse = (actual, predicted) => Math.sqrt(mean(actual.map((value, index) => (value - predicted[index]) ** 2)));
const mae = (actual, predicted) => mean(actual.map((value, index) => Math.abs(value - predicted[index])));
const rSquared = (actual, predicted) => { const baseline = mean(actual); return 1 - actual.reduce((sum, value, i) => sum + (value - predicted[i]) ** 2, 0) / actual.reduce((sum, value) => sum + (value - baseline) ** 2, 0); };
const shuffle = (rows, seed = 42) => { let state = seed; const random = () => (state = (state * 1664525 + 1013904223) >>> 0) / 4294967296; return [...rows].sort(() => random() - .5); };

function linearModel(train, fields, ridge = 0) {
  const xMeans = fields.map((field) => mean(train.map((row) => row[field])));
  const xStds = fields.map((field, index) => std(train.map((row) => row[field])) || 1);
  const yMean = mean(train.map((row) => row.target));
  const weights = Array(fields.length + 1).fill(0);
  const features = (row) => [1, ...fields.map((field, i) => (row[field] - xMeans[i]) / xStds[i])];
  for (let iteration = 0; iteration < 1800; iteration++) {
    const gradient = Array(weights.length).fill(0);
    for (const row of train) { const vector = features(row); const error = vector.reduce((sum, value, i) => sum + value * weights[i], 0) + yMean - row.target; vector.forEach((value, i) => gradient[i] += error * value); }
    weights.forEach((weight, i) => { gradient[i] = gradient[i] / train.length + (i ? ridge * weight / train.length : 0); weights[i] -= .035 * gradient[i]; });
  }
  return (row) => { const vector = features(row); return vector.reduce((sum, value, i) => sum + value * weights[i], 0) + yMean; };
}
function forestModel(train, fields) {
  let state = 84; const random = () => (state = (state * 1103515245 + 12345) >>> 0) / 4294967296; const trees = [];
  for (let t = 0; t < 120; t++) { const sample = Array.from({ length: train.length }, () => train[Math.floor(random() * train.length)]); const field = fields[Math.floor(random() * fields.length)]; const values = sample.map((row) => row[field]); const threshold = quantile(values, .15 + random() * .7); const left = sample.filter((row) => row[field] <= threshold).map((row) => row.target); const right = sample.filter((row) => row[field] > threshold).map((row) => row.target); trees.push({ field, threshold, left: left.length ? mean(left) : mean(sample.map((row) => row.target)), right: right.length ? mean(right) : mean(sample.map((row) => row.target)) }); }
  return (row) => mean(trees.map((tree) => row[tree.field] <= tree.threshold ? tree.left : tree.right));
}
function regression(rows) {
  const fields = ["budget", "battery_life", "storage", "lifespan", "weight", "ram"].filter((field) => rows.some((row) => numeric(row[field]) != null));
  const prepared = rows.map((row) => { const out = { target: numeric(row.device_price) }; fields.forEach((field) => out[field] = numeric(row[field])); return out; }).filter((row) => row.target != null);
  fields.forEach((field) => { const fill = median(prepared.map((row) => row[field]).filter((value) => value != null)); prepared.forEach((row) => { if (row[field] == null) row[field] = fill; }); });
  const shuffled = shuffle(prepared); const split = Math.floor(shuffled.length * .7); const train = shuffled.slice(0, split); const test = shuffled.slice(split);
  const definitions = [{ name: "Linear Regression", fit: (set) => linearModel(set, fields) }, { name: "Ridge Regression", fit: (set) => linearModel(set, fields, 5) }, { name: "Random Forest", fit: (set) => forestModel(set, fields) }];
  const modelRows = definitions.map((definition) => { const model = definition.fit(train); const predicted = test.map(model); const actual = test.map((row) => row.target); const folds = 5; const cv = [];
    for (let fold = 0; fold < folds; fold++) { const validation = train.filter((_, i) => i % folds === fold); const fitting = train.filter((_, i) => i % folds !== fold); const foldModel = definition.fit(fitting); cv.push(rmse(validation.map((row) => row.target), validation.map(foldModel))); }
    return { model: definition.name, cvRmse: round(mean(cv), 2), cvMae: null, cvR2: null, mae: round(mae(actual, predicted), 2), rmse: round(rmse(actual, predicted), 2), r2: round(rSquared(actual, predicted), 3), diagnostics: actual.map((value, i) => ({ actual: round(value), predicted: round(predicted[i]), residual: round(value - predicted[i]) })) };
  });
  const actual = test.map((row) => row.target); const baseline = Array(test.length).fill(mean(train.map((row) => row.target)));
  modelRows.push({ model: "Training-mean benchmark", cvRmse: null, cvMae: null, cvR2: null, mae: round(mae(actual, baseline), 2), rmse: round(rmse(actual, baseline), 2), r2: round(rSquared(actual, baseline), 3), diagnostics: [] });
  const selected = [...modelRows].filter((model) => model.cvRmse != null).sort((a, b) => a.cvRmse - b.cvRmse)[0];
  return { trainingRows: train.length, testRows: test.length, predictors: fields.length, selectedModel: selected.model, models: modelRows.map(({ diagnostics, ...model }) => model), diagnostics: selected.diagnostics };
}

const raw = readWorkbook(rawFile); const clean = readWorkbook(cleanFile);
const columns = (rows) => new Set(rows.flatMap((row) => Object.keys(row)));
const numericFields = ["battery_life", "budget", "device_price", "storage", "lifespan", "weight", "ram"].filter((field) => columns(clean).has(field));
const cleanNumbers = (field) => clean.map((row) => numeric(row[field])).filter((value) => value != null);
const summary = numericFields.map((field) => { const values = cleanNumbers(field); return { variable: field.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()), count: values.length, mean: round(mean(values)), median: round(median(values)), std: round(std(values)), min: round(min(values)), max: round(max(values)) }; });
const priceValues = cleanNumbers("device_price"); const binMin = min(priceValues); const binWidth = (max(priceValues) - binMin) / 8; const priceHistogram = Array.from({ length: 8 }, (_, index) => { const low = binMin + index * binWidth; const high = low + binWidth; return { range: `${Math.round(low / 1000)}k–${Math.round(high / 1000)}k`, count: priceValues.filter((value) => value >= low && (index === 7 ? value <= high : value < high)).length }; });
const categories = Object.fromEntries(["device_type", "brand_name", "primary_influence", "satisfaction", "chosen_method"].filter((key) => columns(clean).has(key)).map((key) => [key, valueCounts(clean, key)]));
const pairs = clean.map((row) => ({ budget: numeric(row.budget), device_price: numeric(row.device_price), deviceType: row.device_type })).filter((row) => row.budget != null && row.device_price != null);
const paired = pairs; const differences = paired.map((row) => row.device_price - row.budget); const t = mean(differences) / (std(differences) / Math.sqrt(differences.length)); const tP = 2 * (1 - normalCdf(Math.abs(t))); const laptopMask = clean.map((row) => /laptop|desktop/i.test(String(row.device_type))); const researchMask = clean.map((row) => /few weeks|few months|^month$/i.test(String(row.research_time)));
const proportionTest = (question, mask) => { const successes = mask.filter(Boolean).length; const proportion = successes / mask.length; const z = (proportion - .5) / Math.sqrt(.25 / mask.length); return { question, test: "One-sample proportion z-test", null: "p = 0.50", alternative: "p > 0.50", sampleSize: mask.length, successes, proportion: round(proportion), statisticLabel: "Z", statistic: round(z), pValue: round(1 - normalCdf(z)), decision: 1 - normalCdf(z) < .05 ? "Reject H₀" : "Fail to reject H₀" }; };
const rawMissing = raw.reduce((sum, row) => sum + Object.values(row).filter((value) => value == null || value === "").length, 0); const cleanMissing = clean.reduce((sum, row) => sum + Object.values(row).filter((value) => value == null || value === "").length, 0);
const rawColumnCount = sourceColumnCount(rawFile); const cleanColumnCount = sourceColumnCount(cleanFile);
const data = { generatedFrom: { raw: path.basename(rawFile), cleaned: path.basename(cleanFile), rawRows: raw.length, cleanRows: clean.length, rawColumns: rawColumnCount, cleanColumns: cleanColumnCount }, overview: { respondents: clean.length, meanBudget: round(mean(cleanNumbers("budget"))), meanPrice: round(mean(priceValues)), medianPrice: round(median(priceValues)), laptopShare: round((categories.device_type || []).find((item) => /laptop|desktop/i.test(item.label))?.percentage ?? 0, 2) }, cleaning: { removedRows: raw.length - clean.length, removedColumns: rawColumnCount - cleanColumnCount, rawMissingCells: rawMissing, cleanMissingCells: cleanMissing, numericFields, steps: ["Standardised survey labels for analysis fields", "Extracted numeric values and aligned measurement units", "Normalised categorical response labels", "Removed the empty survey column and incomplete records"], caveat: "Abbreviated price entries and ambiguous categorical responses remain documented data-quality assumptions." }, descriptive: { summary, categories, priceHistogram, budgetVsPrice: pairs }, regression: regression(clean), inferential: { alpha: .05, tests: [{ question: "Do actual device prices differ from allocated budgets?", test: "Paired t-test", null: "μ difference = 0", alternative: "μ difference ≠ 0", sampleSize: paired.length, meanBudget: round(mean(paired.map((row) => row.budget))), meanPrice: round(mean(paired.map((row) => row.device_price))), meanDifference: round(mean(differences)), statisticLabel: "t", statistic: round(t), pValue: round(tP), decision: tP < .05 ? "Reject H₀" : "Fail to reject H₀" }, proportionTest("Do more than 50% of respondents prefer Laptop/Desktop computers?", laptopMask), proportionTest("Did more than 50% research for over one week?", researchMask)] } };
fs.mkdirSync(path.dirname(outputFile), { recursive: true }); fs.writeFileSync(outputFile, JSON.stringify(data, null, 2)); console.log(`Generated ${outputFile}`);
