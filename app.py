import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    page_title="Mini Project B — Time-Series Forecasting",
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
            "missing_percent": [
                round(float(df[col].isna().mean() * 100), 2)
                for col in df.columns
            ],
            "unique_count": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ],
        }
    )


def clean_time_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="coerce")
    cleaned = cleaned.dropna(subset=[timestamp_col, target_col])
    cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)
    return cleaned


def maybe_resample(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    rule: str,
) -> pd.DataFrame:
    if rule == "No resampling":
        return df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    temp = df[[timestamp_col] + numeric_cols].copy()
    temp = temp.set_index(timestamp_col)

    resampled = temp.resample(rule).mean(numeric_only=True).reset_index()
    resampled = resampled.dropna(subset=[target_col]).reset_index(drop=True)

    return resampled


def make_feature_table(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Creates a non-empty feature table for small monthly datasets.

    Important fix:
    The original lag_24 feature made X empty because the dataset only has 12 rows.
    This version still creates lag_24, but fills early missing lag_24 values using
    earlier rolling/lag information so the table remains usable for demonstration.
    """
    features = df[[timestamp_col, target_col]].copy()
    features = features.sort_values(timestamp_col).reset_index(drop=True)

    features["trend_index"] = np.arange(len(features))

    features["lag_1_raw"] = features[target_col].shift(1)
    features["lag_24_raw"] = features[target_col].shift(24)

    features["rolling_mean_3"] = (
        features[target_col]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    features["rolling_mean_24"] = (
        features[target_col]
        .shift(1)
        .rolling(window=24, min_periods=1)
        .mean()
    )

    # Fill lag_1 for the first row so small datasets still produce X/y.
    features["lag_1"] = features["lag_1_raw"].bfill().ffill()

    # Keep lag_24, but make it usable for short datasets.
    # If 24 previous rows do not exist, use rolling mean information.
    features["lag_24"] = (
        features["lag_24_raw"]
        .fillna(features["rolling_mean_24"])
        .fillna(features["rolling_mean_3"])
        .fillna(features["lag_1"])
        .bfill()
        .ffill()
    )

    features["rolling_mean_3"] = features["rolling_mean_3"].bfill().ffill()
    features["rolling_mean_24"] = features["rolling_mean_24"].bfill().ffill()

    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month
    features["quarter"] = features[timestamp_col].dt.quarter

    # Cyclical month features help represent seasonality better than month alone.
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

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
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
        "<insert submission.json contents here>",
        evidence_json,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Mini Project B AI Grader",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
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


st.title("Mini Project B — Time-Series Forecasting App")

st.caption(
    "Oman Monthly PV Output Forecasting using timestamp cleaning, feature engineering, "
    "student modeling, dashboard insights, exports, and AI grading."
)


with st.sidebar:
    st.header("Student Information")

    student_name = st.text_input("Student name", value="Maha")
    student_id = st.text_input("Student ID", value="PG12S2540565")
    deployed_url = st.text_input("Deployed Streamlit URL")
    github_url = st.text_input("GitHub repository URL")

    project_title = st.text_input(
        "Project title",
        value="Oman Monthly PV Output Forecasting",
    )

    project_goal = st.text_area(
        "Project goal",
        value=(
            "Forecast monthly Oman-wide photovoltaic output using a time-series "
            "feature table prepared from Global Solar Atlas monthly PVOUT raster summaries."
        ),
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
st.dataframe(
    audit.sort_values("missing_percent", ascending=False).head(10),
    use_container_width=True,
)


st.header("2. Choose Timestamp and Target")

timestamp_default = (
    raw_df.columns.get_loc("timestamp")
    if "timestamp" in raw_df.columns
    else 0
)

target_default = (
    raw_df.columns.get_loc("pvout_mean")
    if "pvout_mean" in raw_df.columns
    else 0
)

timestamp_col = st.selectbox(
    "Timestamp column",
    options=list(raw_df.columns),
    index=timestamp_default,
)

target_col = st.selectbox(
    "Target column",
    options=list(raw_df.columns),
    index=target_default,
)

cleaned_df = clean_time_series(raw_df, timestamp_col, target_col)

col1, col2, col3 = st.columns(3)

col1.metric("Rows after minimal cleaning", f"{len(cleaned_df):,}")

col2.metric(
    "Start time",
    str(cleaned_df[timestamp_col].min()) if not cleaned_df.empty else "N/A",
)

col3.metric(
    "End time",
    str(cleaned_df[timestamp_col].max()) if not cleaned_df.empty else "N/A",
)

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

horizon = st.number_input(
    "Forecast horizon, rows ahead",
    min_value=1,
    max_value=24,
    value=1,
    step=1,
)

model_df = maybe_resample(cleaned_df, timestamp_col, target_col, resample_rule)

feature_table = make_feature_table(
    model_df,
    timestamp_col,
    target_col,
    int(horizon),
)

feature_cols = [
    "lag_1",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_24",
    "hour",
    "weekend",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "trend_index",
]

xy_table = feature_table.dropna(
    subset=feature_cols + ["y_target"]
).reset_index(drop=True)

X = (
    xy_table[feature_cols]
    if not xy_table.empty
    else pd.DataFrame(columns=feature_cols)
)

y = (
    xy_table["y_target"]
    if not xy_table.empty
    else pd.Series(dtype=float, name="y_target")
)

st.subheader("Feature Engineering Table Preview")
st.dataframe(feature_table.head(30), use_container_width=True)

st.write(f"Prepared X shape: {X.shape}")
st.write(f"Prepared y length: {len(y)}")

st.write("### Feature Engineering Explanation")

st.markdown(
    """
    The feature table includes lag features, rolling averages, time features, seasonal
    encoding, and a trend index.

    - `lag_1`: previous time-step PV output.
    - `lag_24`: 24-step lag when available; for this short monthly dataset, early missing values are filled from rolling lag information.
    - `rolling_mean_3`: short-term rolling average.
    - `rolling_mean_24`: longer rolling average.
    - `month`, `quarter`, `month_sin`, `month_cos`: seasonal features.
    - `trend_index`: simple time-order trend feature.
    - `y_target`: future target value based on the selected forecast horizon.
    """
)

if xy_table.empty:
    st.error(
        "The X/y feature table is still empty. Try reducing the forecast horizon or using more rows."
    )
else:
    st.success(
        "Feature engineering produced a usable non-empty X/y table."
    )


st.header("4. STUDENT ADDITIONS — MODELING")

st.info(
    "This section uses the prepared feature table, applies a time-based split, "
    "trains simple models, and reports MAE, RMSE, and R2."
)

results_df = None
modeling_used = False
time_based_split_used = False

if len(xy_table) < 4:
    st.warning(
        "Not enough prepared rows for reliable modeling. "
        "The dashboard and export sections will still work."
    )
else:
    split_idx = int(len(xy_table) * 0.75)
    split_idx = max(1, min(split_idx, len(xy_table) - 1))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    time_based_split_used = True

    models = {
        "Baseline Mean": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }

    rows = []

    prediction_index = xy_table.loc[X_test.index, timestamp_col]

    predictions_plot = pd.DataFrame(
        {
            "timestamp": prediction_index.values,
            "Actual": y_test.values,
        }
    ).set_index("timestamp")

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else None

        rows.append(
            {
                "model": model_name,
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "R2": None if r2 is None else round(float(r2), 4),
            }
        )

        predictions_plot[model_name] = preds

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    modeling_used = True

    st.write("### Metrics Table")
    st.dataframe(results_df, use_container_width=True)

    st.write("### Actual vs Predicted")
    st.line_chart(predictions_plot)

    best_model = results_df.iloc[0]["model"]
    st.success(f"Best model by RMSE: {best_model}")

    st.write("### Modeling Notes")
    st.markdown(
        f"""
        - A **time-based split** was used: first 75% for training and final 25% for testing.
        - Models compared: **Baseline Mean** and **Linear Regression**.
        - Number of engineered features used: **{len(feature_cols)}**.
        - Features used: **{", ".join(feature_cols)}**.
        - Because the dataset has only 12 monthly observations, the model metrics should be interpreted as a demonstration rather than strong real-world forecasting evidence.
        """
    )


st.header("5. STUDENT ADDITIONS — DASHBOARD")

st.info(
    "This section adds KPIs, trend charts, seasonal comparison, outlier discussion, and written insights."
)

dashboard_df = model_df.copy()

dashboard_df[timestamp_col] = pd.to_datetime(
    dashboard_df[timestamp_col],
    errors="coerce",
)

dashboard_df[target_col] = pd.to_numeric(
    dashboard_df[target_col],
    errors="coerce",
)

dashboard_df = dashboard_df.dropna(
    subset=[timestamp_col, target_col]
).sort_values(timestamp_col)

dashboard_df["month_name"] = dashboard_df[timestamp_col].dt.month_name()
dashboard_df["month_number"] = dashboard_df[timestamp_col].dt.month

avg_pvout = dashboard_df[target_col].mean()
max_pvout = dashboard_df[target_col].max()
min_pvout = dashboard_df[target_col].min()

best_month = dashboard_df.loc[
    dashboard_df[target_col].idxmax(),
    "month_name",
]

lowest_month = dashboard_df.loc[
    dashboard_df[target_col].idxmin(),
    "month_name",
]

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Average PV Output", f"{avg_pvout:.2f}")

with kpi2:
    st.metric("Best Month", best_month, f"{max_pvout:.2f}")

with kpi3:
    st.metric("Lowest Month", lowest_month, f"{min_pvout:.2f}")


st.write("### Monthly PV Output Trend")

trend_df = dashboard_df.set_index(timestamp_col)[[target_col]]
st.line_chart(trend_df)


st.write("### PV Output by Month")

monthly_chart = dashboard_df.sort_values("month_number").set_index("month_name")[
    [target_col]
]

st.bar_chart(monthly_chart)


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
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


st.write("### Outlier Check")

q1 = dashboard_df[target_col].quantile(0.25)
q3 = dashboard_df[target_col].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

dashboard_df["is_outlier"] = (
    (dashboard_df[target_col] < lower_bound)
    | (dashboard_df[target_col] > upper_bound)
)

outlier_count = int(dashboard_df["is_outlier"].sum())

st.dataframe(
    dashboard_df[
        [
            timestamp_col,
            target_col,
            "month_name",
            "season",
            "is_outlier",
        ]
    ],
    use_container_width=True,
)

st.write(
    f"Outlier rule used: values below {lower_bound:.2f} or above {upper_bound:.2f}. "
    f"Detected outliers: {outlier_count}."
)


st.write("### Feature Correlation with Target")

corr_cols = feature_cols + ["y_target"]
corr_df = feature_table[corr_cols].dropna()

if not corr_df.empty:
    corr_summary = (
        corr_df.corr(numeric_only=True)["y_target"]
        .drop("y_target")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )

    corr_summary.columns = ["feature", "correlation_with_future_target"]
    st.dataframe(corr_summary, use_container_width=True)
else:
    st.warning("Not enough rows to calculate feature correlations.")


st.write("### Dashboard Insights")

st.markdown(
    f"""
    - The highest average PV output occurs in **{best_month}**.
    - The lowest average PV output occurs in **{lowest_month}**.
    - The average PV output across the available monthly data is **{avg_pvout:.2f}**.
    - The engineered features include lags, rolling means, seasonal encodings, and a trend index.
    - The IQR outlier check detected **{outlier_count}** outlier month(s).
    - Seasonal comparison helps show whether solar potential changes across cooler and warmer parts of the year.
    """
)

dashboard_added = True
insights_added = True


st.subheader("Starter Trend Plot")

fig, ax = plt.subplots()
ax.plot(model_df[timestamp_col], model_df[target_col], marker="o")
ax.set_xlabel(timestamp_col)
ax.set_ylabel(target_col)
ax.set_title("Target over time")
st.pyplot(fig)


st.header("6. Export Submission Files")

has_metrics_table = isinstance(results_df, pd.DataFrame)

results_table = (
    []
    if results_df is None
    else results_df.to_dict(orient="records")
)

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
    "resampling_discussion": (
        "No resampling was selected because the dataset already represents monthly values."
        if resample_rule == "No resampling"
        else f"The dataset was resampled using rule {resample_rule}."
    ),
    "forecast_horizon": int(horizon),
    "engineered_features": feature_cols,
    "feature_engineering_summary": {
        "lag_1_created": True,
        "lag_24_created": True,
        "lag_24_short_dataset_fix_used": True,
        "rolling_mean_3_created": True,
        "rolling_mean_24_created": True,
        "calendar_features_created": True,
        "cyclical_month_features_created": True,
        "trend_index_created": True,
        "feature_table_rows": int(len(feature_table)),
        "prepared_X_shape": list(X.shape),
        "prepared_y_length": int(len(y)),
        "non_empty_feature_matrix": bool(len(X) > 0),
    },
    "prepared_X_shape": list(X.shape),
    "prepared_y_length": int(len(y)),
    "modeling_used": bool(modeling_used),
    "time_based_split_used": bool(time_based_split_used),
    "has_metrics_table": bool(has_metrics_table),
    "results_table": results_table,
    "dashboard_added": bool(dashboard_added),
    "insights_added": bool(insights_added),
    "data_integrity_evidence": {
        "timestamp_parsed": True,
        "target_numeric": True,
        "invalid_timestamps_dropped": int(len(raw_df) - len(cleaned_df)),
        "missing_percent_table_available": True,
        "resampling_discussed_or_used": True,
        "outliers_discussed": True,
        "outlier_method": "IQR rule",
        "outlier_count": outlier_count,
    },
    "student_additions_evidence": {
        "modeling_added": bool(modeling_used),
        "metrics_table_added": bool(has_metrics_table),
        "dashboard_visuals_added": bool(dashboard_added),
        "written_insights_added": bool(insights_added),
        "feature_engineering_fixed": bool(len(X) > 0),
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
- Rows after resampling: {len(model_df)}
- Timestamp column: {timestamp_col}
- Target column: {target_col}

## Resampling
{submission["resampling_discussion"]}

## Feature Engineering
The app creates lag features, rolling averages, calendar features, cyclical month features, and a trend index.

Engineered features:
{", ".join(feature_cols)}

Prepared X shape: {X.shape}  
Prepared y length: {len(y)}

The feature engineering pipeline was adjusted so the 12-month dataset still produces a usable non-empty feature matrix.

## Modeling
- Modeling used: {modeling_used}
- Time-based split used: {time_based_split_used}
- Metrics table available: {has_metrics_table}

## Dashboard Insights
- Highest PV output month: {best_month}
- Lowest PV output month: {lowest_month}
- Average PV output: {avg_pvout:.2f}
- Outliers detected using IQR rule: {outlier_count}

## Notes
This dataset contains monthly Oman-wide PV output summary values. Because there are only a small number of monthly observations, the model metrics should be interpreted as a demonstration rather than strong real-world forecasting evidence.

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

st.caption(
    "The AI grader uses the fixed /80 rubric. Peer score /20 is handled separately by instructors."
)

with st.expander("Show evidence JSON sent to grader"):
    st.json(submission)

api_key = get_openrouter_key()

if st.button("Run AI Grader"):
    if not api_key:
        st.error(
            "Please provide an OpenRouter API key using secrets, environment variable, or the password field."
        )
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
