import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def get_openrouter_key():
    """Read API key from Streamlit secrets, environment, or UI input."""
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    return st.text_input("OpenRouter API key", type="password")


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
            "unique_count": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def clean_time_series(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col])
    cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)
    return cleaned


def maybe_resample(df: pd.DataFrame, timestamp_col: str, target_col: str, rule: str) -> pd.DataFrame:
    if rule == "No resampling":
        return df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    temp = df[[timestamp_col] + numeric_cols].copy()
    temp = temp.set_index(timestamp_col)
    resampled = temp.resample(rule).mean(numeric_only=True).reset_index()
    return resampled.dropna(subset=[target_col]).reset_index(drop=True)


def make_baseline_features(df: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    features = df[[timestamp_col, target_col]].copy()
    features = features.sort_values(timestamp_col).reset_index(drop=True)

    features["lag_1"] = features[target_col].shift(1)
    features["lag_24"] = features[target_col].shift(24)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(window=24, min_periods=1).mean()

    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month

    features["y_target"] = features[target_col].shift(-horizon)
    return features


def parse_grader_response(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def call_ai_grader(api_key: str, evidence_json: str):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", evidence_json)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Mini Project B AI Grader",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("Starter app stops at data audit, feature-table preparation, exports, and AI grading. Add your own modeling, metrics, and dashboard work.")

with st.sidebar:
    st.header("Student Information")
    student_name = st.text_input("Student name", value="Maha")
    student_id = st.text_input("Student ID", value="PG12S2540565")
    deployed_url = st.text_input("Deployed Streamlit URL")
    github_url = st.text_input("GitHub repository URL")
    project_title = st.text_input("Project title", value="Oman Monthly PV Output Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast monthly Oman-wide photovoltaic output using a time-series feature table prepared from Global Solar Atlas monthly PVOUT raster summaries.",
        height=100,
    )

st.header("1. Load Local Dataset")
dataset_path = st.text_input("Dataset path", value="data/dataset_sample.csv")

try:
    raw_df = load_dataset(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.subheader("First 10 Rows")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("Dataset Audit")
audit = audit_dataframe(raw_df)
st.dataframe(audit, use_container_width=True)

st.subheader("Missing Percent — Top 10")
st.dataframe(audit.sort_values("missing_percent", ascending=False).head(10), use_container_width=True)

st.header("2. Choose Timestamp and Target")
timestamp_default = raw_df.columns.get_loc("timestamp") if "timestamp" in raw_df.columns else 0
numeric_candidates = raw_df.select_dtypes(include=[np.number]).columns.tolist()
target_default = raw_df.columns.get_loc("pvout_mean") if "pvout_mean" in raw_df.columns else 0

timestamp_col = st.selectbox("Timestamp column", options=list(raw_df.columns), index=timestamp_default)
target_col = st.selectbox("Target column", options=list(raw_df.columns), index=target_default)

cleaned_df = clean_time_series(raw_df, timestamp_col, target_col)

col1, col2, col3 = st.columns(3)
col1.metric("Rows after minimal cleaning", f"{len(cleaned_df):,}")
col2.metric("Start time", str(cleaned_df[timestamp_col].min()) if not cleaned_df.empty else "N/A")
col3.metric("End time", str(cleaned_df[timestamp_col].max()) if not cleaned_df.empty else "N/A")

if cleaned_df.empty:
    st.error("No valid rows remain after parsing timestamps and target values.")
    st.stop()

st.header("3. Optional Resampling and Forecast Horizon")
resample_rule = st.selectbox(
    "Resampling option",
    options=["No resampling", "D", "W", "MS", "QS", "YS"],
    index=0,
    help="D=daily, W=weekly, MS=month start, QS=quarter start, YS=year start.",
)
horizon = st.number_input("Forecast horizon (rows ahead)", min_value=1, max_value=24, value=1, step=1)

model_df = maybe_resample(cleaned_df, timestamp_col, target_col, resample_rule)
feature_table = make_baseline_features(model_df, timestamp_col, target_col, int(horizon))

baseline_feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
xy_table = feature_table.dropna(subset=baseline_feature_cols + ["y_target"]).reset_index(drop=True)
X = xy_table[baseline_feature_cols] if not xy_table.empty else pd.DataFrame(columns=baseline_feature_cols)
y = xy_table["y_target"] if not xy_table.empty else pd.Series(dtype=float, name="y_target")

st.subheader("Baseline Feature Table Preview")
st.dataframe(feature_table.head(30), use_container_width=True)

st.write(f"Prepared X shape: {X.shape}")
st.write(f"Prepared y length: {len(y)}")

if xy_table.empty:
    st.warning("The prepared X/y table is empty. This can happen when the dataset is short and lag_24 is required. You may need more data before training.")

st.header("4. STUDENT ADDITIONS — MODELING")
st.info("Add your own time-based split, models, predictions, metrics table, and error analysis below this marker in app.py.")
st.code(
    """
# STUDENT ADDITIONS - MODELING
# ===============================
# STUDENT ADDITIONS — MODELING
# Paste this under the MODELING marker
# ===============================

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results_df = None

st.subheader("Student Addition: Forecasting Models")

if len(feature_df) < 8:
    st.warning("Not enough rows for a reliable train/test split. Add more time periods if available.")
else:
    # Time-based split: first 75% train, last 25% test
    split_idx = int(len(feature_df) * 0.75)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    models = {
        "Baseline Mean": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=3
        )
    }

    rows = []
    predictions_plot = pd.DataFrame({
        "actual": y_test.values
    }, index=y_test.index)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else None

        rows.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        predictions_plot[model_name] = preds

    results_df = pd.DataFrame(rows).sort_values("RMSE")

    st.write("### Metrics Table")
    st.dataframe(results_df, use_container_width=True)

    st.write("### Actual vs Predicted")
    st.line_chart(predictions_plot)

    best_model = results_df.iloc[0]["model"]
    st.success(f"Best model by RMSE: {best_model}")

    st.info(
        "Because this dataset has only 12 monthly observations, the metrics are useful "
        "as a demonstration but should not be treated as strong evidence of real-world accuracy."
# Train at least one meaningful forecasting model.
# Create a metrics table named results_df.
# Example final object expected by exports:
# results_df = pd.DataFrame([...])
""",
    language="python",
)

results_df = None

st.header("5. STUDENT ADDITIONS — DASHBOARD")
st.info("Add your own dashboard visuals, KPIs, forecast-vs-actual plots, and written insights below this marker in app.py.")
st.code(
    """
# STUDENT ADDITIONS - DASHBOARD
# ===============================
# STUDENT ADDITIONS — DASHBOARD
# Paste this under the DASHBOARD marker
# ===============================

st.subheader("Student Addition: Solar PV Dashboard")

dashboard_df = ts_df.copy()

# Ensure timestamp column is datetime
dashboard_df[time_col] = pd.to_datetime(dashboard_df[time_col], errors="coerce")
dashboard_df = dashboard_df.dropna(subset=[time_col, target_col]).sort_values(time_col)

dashboard_df["month_name"] = dashboard_df[time_col].dt.month_name()
dashboard_df["month_number"] = dashboard_df[time_col].dt.month

# KPI row
avg_pvout = dashboard_df[target_col].mean()
max_pvout = dashboard_df[target_col].max()
min_pvout = dashboard_df[target_col].min()
best_month = dashboard_df.loc[dashboard_df[target_col].idxmax(), "month_name"]
lowest_month = dashboard_df.loc[dashboard_df[target_col].idxmin(), "month_name"]

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Average PV Output", f"{avg_pvout:.2f}")

with kpi2:
    st.metric("Best Month", best_month, f"{max_pvout:.2f}")

with kpi3:
    st.metric("Lowest Month", lowest_month, f"{min_pvout:.2f}")

# Trend chart
st.write("### Monthly PV Output Trend")
trend_df = dashboard_df.set_index(time_col)[[target_col]]
st.line_chart(trend_df)

# Monthly bar chart
st.write("### PV Output by Month")
monthly_chart = dashboard_df.sort_values("month_number").set_index("month_name")[[target_col]]
st.bar_chart(monthly_chart)

# Seasonal comparison
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

dashboard_df["season"] = dashboard_df["month_number"].apply(get_season)

season_df = (
    dashboard_df.groupby("season", as_index=False)[target_col]
    .mean()
    .sort_values(target_col, ascending=False)
)

st.write("### Average PV Output by Season")
st.dataframe(season_df, use_container_width=True)
st.bar_chart(season_df.set_index("season")[[target_col]])

# Insight text
st.write("### Dashboard Insights")
st.markdown(
    f"""
    - The highest average PV output occurs in **{best_month}**.
    - The lowest average PV output occurs in **{lowest_month}**.
    - The average PV output across the available monthly data is **{avg_pvout:.2f}**.
    - Seasonal comparison helps show whether solar potential is stronger in cooler or warmer parts of the year.
    """
)
    )
""",
    language="python",
)

st.subheader("Starter Trend Plot")
fig, ax = plt.subplots()
ax.plot(model_df[timestamp_col], model_df[target_col], marker="o")
ax.set_xlabel(timestamp_col)
ax.set_ylabel(target_col)
ax.set_title("Target over time")
st.pyplot(fig)

st.header("6. Export Submission Files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

submission = {
    "student_name": student_name,
    "student_id": student_id,
    "project_title": project_title,
    "project_goal": project_goal,
    "deployed_url": deployed_url,
    "github_url": github_url,
    "timestamp_column": timestamp_col,
    "target_column": target_col,
    "rows_raw": int(len(raw_df)),
    "rows_after_cleaning": int(len(cleaned_df)),
    "rows_after_resampling": int(len(model_df)),
    "resampling": resample_rule,
    "forecast_horizon": int(horizon),
    "baseline_features": baseline_feature_cols,
    "prepared_X_shape": list(X.shape),
    "prepared_y_length": int(len(y)),
    "has_metrics_table": bool(has_metrics_table),
    "results_table": results_table,
    "data_integrity_evidence": {
        "timestamp_parsed": True,
        "target_numeric": True,
        "invalid_timestamps_dropped": int(len(raw_df) - len(cleaned_df)),
        "missing_percent_table_available": True,
        "resampling_discussed_or_used": resample_rule != "No resampling",
        "outliers_discussed": False,
    },
    "student_additions_expected": {
        "modeling_required": True,
        "dashboard_required": True,
        "insights_required": True,
    },
    "generated_at": datetime.utcnow().isoformat() + "Z",
}

submission_json = json.dumps(submission, indent=2)

project_card = f"""# {project_title}

## Student
- Name: {student_name}
- ID: {student_id}

## Goal
{project_goal}

## Dataset
- Rows raw: {len(raw_df)}
- Rows after cleaning: {len(cleaned_df)}
- Timestamp column: {timestamp_col}
- Target column: {target_col}

## Feature Table
Baseline starter features prepared:
{", ".join(baseline_feature_cols)}

Prepared X shape: {X.shape}  
Prepared y length: {len(y)}

## Student Work Still Required
Add your own modeling, metrics table, dashboard visuals, and written insights under the STUDENT ADDITIONS markers in app.py.

## Links
- Streamlit app: {deployed_url}
- GitHub repo: {github_url}
"""

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download submission.json",
        data=submission_json,
        file_name="submission.json",
        mime="application/json",
    )
with col_b:
    st.download_button(
        "Download project_card.md",
        data=project_card,
        file_name="project_card.md",
        mime="text/markdown",
    )

st.header("7. AI Grader (/80)")
st.caption("The AI grader uses the fixed /80 rubric. Peer score /20 is handled separately by instructors.")

with st.expander("Show evidence JSON sent to grader"):
    st.json(submission)

api_key = get_openrouter_key()

if st.button("Run AI Grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key using secrets, environment variable, or the password field.")
    else:
        try:
            raw_output = call_ai_grader(api_key, submission_json)
            parsed = parse_grader_response(raw_output)
            if parsed is not None:
                st.subheader("Parsed AI Grader JSON")
                st.json(parsed)
            else:
                st.subheader("Raw AI Grader Output")
                st.text(raw_output)
        except Exception as exc:
            st.error(f"AI grader failed: {exc}")
