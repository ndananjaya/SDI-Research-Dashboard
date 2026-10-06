"""Build the public dashboard data from the two Excel datasets.

Run from the dashboard folder:
  ..\\SDI_Assignment\\.venv-dashboard\\bin\\python.exe scripts\\generate_dashboard_data.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_key(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").strip().lower().split())


ALIASES = {
    "timestamp": ["timestamp"],
    "faculty": ["faculty"],
    "device_type": ["device_type", "preferred device type"],
    "brand_name": ["brand_name", "brand name"],
    "chosen_method": ["chosen_method", "how did you choose"],
    "battery_life": ["normal_use_battery_life", "battery life time"],
    "budget": ["budget", "allocated budget"],
    "device_price": ["device_price", "actual price"],
    "payment_method": ["payment_method", "payment method"],
    "storage": ["storage_capacity", "total storage capacity"],
    "lifespan": ["expected_life_span", "expected device lifespan"],
    "research_time": ["research_time", "research time"],
    "primary_influence": ["primary_influence", "primary factor influencing"],
    "weight": ["weight(g)", "weight of the device"],
    "store_type": ["store_type", "where did you purchase"],
    "satisfaction": ["satisfaction", "are you satisfied"],
    "ram": ["ram_capacity", "ram size"],
}


def canonicalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    indexed = {clean_key(column): column for column in frame.columns}
    rename = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            found = next((original for key, original in indexed.items() if candidate in key), None)
            if found:
                rename[found] = canonical
                break
    return frame.rename(columns=rename)


def number(value: object) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def counts(frame: pd.DataFrame, column: str) -> list[dict]:
    series = frame[column].fillna("Not recorded").astype(str)
    table = series.value_counts().rename_axis("label").reset_index(name="count")
    table["percentage"] = table["count"] / len(frame) * 100
    return [{"label": r.label, "count": int(r.count), "percentage": round(float(r.percentage), 2)} for r in table.itertuples()]


def summary(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    output = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        output.append({
            "variable": column.replace("_", " ").title(),
            "count": int(values.count()),
            "mean": number(values.mean()),
            "median": number(values.median()),
            "std": number(values.std()),
            "min": number(values.min()),
            "max": number(values.max()),
        })
    return output


def box_stats(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for faculty, group in frame.groupby("faculty", observed=True):
        values = pd.to_numeric(group["device_price"], errors="coerce").dropna()
        if len(values):
            rows.append({"faculty": str(faculty), "min": number(values.min()), "q1": number(values.quantile(.25)), "median": number(values.median()), "q3": number(values.quantile(.75)), "max": number(values.max()), "count": int(len(values))})
    return rows


def inferential(frame: pd.DataFrame) -> list[dict]:
    alpha = .05
    budget = pd.to_numeric(frame["budget"], errors="coerce")
    price = pd.to_numeric(frame["device_price"], errors="coerce")
    paired = pd.concat([budget, price], axis=1).dropna()
    t_stat, p_value = stats.ttest_rel(paired.iloc[:, 1], paired.iloc[:, 0])
    laptop = frame["device_type"].astype(str).str.contains("laptop|desktop", case=False, regex=True)
    research = frame["research_time"].astype(str).str.lower().str.contains("few weeks|few months|^month$", regex=True)

    def proportion_result(question: str, mask: pd.Series, alternative: str) -> dict:
        n = int(mask.notna().sum())
        successes = int(mask.sum())
        observed = successes / n
        z = (observed - .5) / np.sqrt(.25 / n)
        p = 1 - stats.norm.cdf(z)
        return {"question": question, "test": "One-sample proportion z-test", "null": "p = 0.50", "alternative": alternative, "sampleSize": n, "successes": successes, "proportion": round(observed, 4), "statisticLabel": "Z", "statistic": round(float(z), 4), "pValue": round(float(p), 4), "decision": "Reject H₀" if p < alpha else "Fail to reject H₀"}

    return [
        {"question": "Do actual device prices differ from allocated budgets?", "test": "Paired t-test", "null": "μ difference = 0", "alternative": "μ difference ≠ 0", "sampleSize": int(len(paired)), "meanBudget": number(paired.iloc[:, 0].mean()), "meanPrice": number(paired.iloc[:, 1].mean()), "meanDifference": number((paired.iloc[:, 1] - paired.iloc[:, 0]).mean()), "statisticLabel": "t", "statistic": round(float(t_stat), 4), "pValue": round(float(p_value), 4), "decision": "Reject H₀" if p_value < alpha else "Fail to reject H₀"},
        proportion_result("Do more than 50% of respondents prefer Laptop/Desktop computers?", laptop, "p > 0.50"),
        proportion_result("Did more than 50% research for over one week?", research, "p > 0.50"),
    ]


def regression(frame: pd.DataFrame) -> dict:
    candidate_columns = [column for column in frame.columns if column not in {"timestamp", "device_price"}]
    target = pd.to_numeric(frame["device_price"], errors="coerce")
    data = frame.loc[target.notna(), candidate_columns].copy()
    target = target.loc[target.notna()]
    numeric = [column for column in candidate_columns if pd.api.types.is_numeric_dtype(data[column]) and column != "satisfaction"]
    categorical = [column for column in candidate_columns if column not in numeric]
    for column in categorical:
        data[column] = data[column].fillna("Not recorded").astype(str)
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(data[column].median())
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.30, random_state=42)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    specs = {
        "Linear Regression": (LinearRegression(), {"selection__percentile": [50, 75, 100]}),
        "Ridge Regression": (Ridge(), {"selection__percentile": [50, 75, 100], "model__alpha": [1., 5., 10.]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1), {"selection__percentile": [50, 75, 100], "model__min_samples_leaf": [2, 3, 4, 5]}),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    output, searches = [], {}
    for name, (model, parameters) in specs.items():
        pipeline = Pipeline([("preprocessing", preprocessor), ("selection", SelectPercentile(f_regression)), ("model", model)])
        search = GridSearchCV(pipeline, parameters, scoring=scores, refit="rmse", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        searches[name] = search
        predicted = search.best_estimator_.predict(x_test)
        index = search.best_index_
        cv_results = search.cv_results_
        output.append({"model": name, "cvRmse": round(float(-cv_results["mean_test_rmse"][index]), 2), "cvMae": round(float(-cv_results["mean_test_mae"][index]), 2), "cvR2": round(float(cv_results["mean_test_r2"][index]), 3), "mae": round(float(mean_absolute_error(y_test, predicted)), 2), "rmse": round(float(np.sqrt(mean_squared_error(y_test, predicted))), 2), "r2": round(float(r2_score(y_test, predicted)), 3), "settings": {key.replace("model__", "").replace("selection__", ""): value for key, value in search.best_params_.items()}})
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    output.append({"model": "Training-mean benchmark", "cvRmse": None, "cvMae": None, "cvR2": None, "mae": round(float(mean_absolute_error(y_test, baseline)), 2), "rmse": round(float(np.sqrt(mean_squared_error(y_test, baseline))), 2), "r2": round(float(r2_score(y_test, baseline)), 3), "settings": {}})
    selected = min(output[:3], key=lambda item: item["cvRmse"])
    selected_model = searches[selected["model"]].best_estimator_
    prediction = selected_model.predict(x_test)
    diagnostics = [{"actual": number(actual), "predicted": number(pred), "residual": number(actual - pred)} for actual, pred in zip(y_test, prediction)]
    return {"trainingRows": int(len(x_train)), "testRows": int(len(x_test)), "predictors": len(candidate_columns), "selectedModel": selected["model"], "models": output, "diagnostics": diagnostics}


def build(raw_path: Path, clean_path: Path, output: Path) -> None:
    raw = canonicalise(pd.read_excel(raw_path))
    clean = canonicalise(pd.read_excel(clean_path))
    required = {"budget", "device_price", "device_type", "research_time", "faculty"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Cleaned dataset is missing required fields: {', '.join(sorted(missing))}")
    numeric = [column for column in ["battery_life", "budget", "device_price", "storage", "lifespan", "weight", "ram"] if column in clean]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    price_values = clean["device_price"].dropna()
    histogram, edges = np.histogram(price_values, bins=8)
    hist = [{"range": f"{int(edges[i]/1000)}k–{int(edges[i + 1]/1000)}k", "count": int(histogram[i])} for i in range(len(histogram))]
    category_data = {column: counts(clean, column) for column in ["device_type", "brand_name", "primary_influence", "satisfaction", "chosen_method"] if column in clean}
    correlations = clean[numeric].corr(numeric_only=True).round(3).replace({np.nan: None}).to_dict()
    payload = {
        "generatedFrom": {"raw": raw_path.name, "cleaned": clean_path.name, "rawRows": int(len(raw)), "cleanRows": int(len(clean)), "rawColumns": int(raw.shape[1]), "cleanColumns": int(clean.shape[1])},
        "overview": {"respondents": int(len(clean)), "meanBudget": number(clean["budget"].mean()), "meanPrice": number(clean["device_price"].mean()), "medianPrice": number(clean["device_price"].median()), "laptopShare": next((item["percentage"] for item in category_data["device_type"] if "laptop" in item["label"].lower() or "desktop" in item["label"].lower()), 0)},
        "cleaning": {"removedRows": int(len(raw) - len(clean)), "removedColumns": int(raw.shape[1] - clean.shape[1]), "rawMissingCells": int(raw.isna().sum().sum()), "cleanMissingCells": int(clean.isna().sum().sum()), "numericFields": numeric, "steps": ["Standardised survey labels for analysis fields", "Extracted numeric values and aligned measurement units", "Normalised categorical response labels", "Removed the empty survey column and incomplete records"], "caveat": "Abbreviated price entries and ambiguous categorical responses should remain documented as data-quality assumptions."},
        "descriptive": {"summary": summary(clean, numeric), "categories": category_data, "priceHistogram": hist, "budgetVsPrice": records(clean[["budget", "device_price", "device_type"]].rename(columns={"device_type": "deviceType"})), "facultyPrice": box_stats(clean), "correlations": correlations},
        "regression": regression(clean),
        "inferential": {"alpha": .05, "tests": inferential(clean)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    dashboard_root = Path(__file__).resolve().parents[1]
    default_data_root = dashboard_root.parent / "SDI_Assignment" / "datasets"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=default_data_root / "uncleaned_response_dataset.xlsx")
    parser.add_argument("--clean", type=Path, default=default_data_root / "cleaned_response_dataset.xlsx")
    parser.add_argument("--output", type=Path, default=dashboard_root / "public" / "data" / "dashboard.json")
    arguments = parser.parse_args()
    build(arguments.raw, arguments.clean, arguments.output)
