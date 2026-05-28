from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from src.data_generator import DataGenerator
from src.model_profiler import DataProfiler
from src.model_selector import AdaptiveModelSelector, validate_model_params
from src.pipeline_config import (
    ModelConfig,
    PipelineConfig,
    get_accurate_config,
    get_default_config,
    get_fast_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/tabular")
OUTPUTS_DIR = Path("outputs")
MAX_ROWS = 500_000
MAX_UPLOAD_MB = 50
SHOW_TRACEBACK_IN_UI = False

APP_VERSION = "3.0.0"
APP_YEAR = 2026
APP_AUTHOR = "Fazbear · Eltex/РиМ/Astra Linux practicum"

st.set_page_config(
    page_title="Astra ML",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --astra-bg: #0b0f1a;
    --astra-surface: #131a2b;
    --astra-surface-2: #1a2238;
    --astra-border: rgba(148, 163, 184, 0.15);
    --astra-accent: #7c3aed;
    --astra-accent-2: #2dd4bf;
    --astra-text: #e6e9f2;
    --astra-muted: #94a3b8;
    --astra-success: #22c55e;
    --astra-warn: #f59e0b;
    --astra-error: #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(124,58,237,0.18), transparent 60%),
                radial-gradient(900px 500px at -10% 20%, rgba(45,212,191,0.12), transparent 60%),
                var(--astra-bg) !important;
    color: var(--astra-text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1322 0%, #0b0f1a 100%) !important;
    border-right: 1px solid var(--astra-border);
}

[data-testid="stSidebar"] * {
    color: var(--astra-text) !important;
}

.astra-hero {
    padding: 36px 44px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(124,58,237,0.28), rgba(45,212,191,0.14));
    border: 1px solid var(--astra-border);
    margin-bottom: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.astra-hero::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(124,58,237,0.18), transparent 70%);
    pointer-events: none;
}
.astra-hero-title {
    font-size: 3.4rem !important;
    line-height: 1.05 !important;
    margin: 0 0 10px 0 !important;
    background: linear-gradient(90deg, #c4b5fd 0%, #818cf8 45%, #5eead4 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    color: transparent !important;
    font-weight: 800;
    letter-spacing: -0.03em;
    display: inline-block;
}
.astra-hero-sub {
    margin: 0;
    color: #cbd5e1;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 720px;
}

.astra-stepper {
    display: flex;
    gap: 8px;
    margin: 14px 0 24px 0;
    padding: 8px;
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 14px;
}
.astra-step {
    flex: 1;
    padding: 10px 14px;
    border-radius: 10px;
    background: transparent;
    color: var(--astra-muted);
    font-size: 0.9rem;
    text-align: center;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}
.astra-step.done {
    color: var(--astra-accent-2);
    border-color: rgba(45,212,191,0.25);
    background: rgba(45,212,191,0.07);
}
.astra-step.active {
    color: #fff;
    background: linear-gradient(135deg, rgba(124,58,237,0.55), rgba(45,212,191,0.35));
    border-color: rgba(124,58,237,0.5);
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(124,58,237,0.25);
}

.astra-card {
    padding: 18px 20px;
    border-radius: 14px;
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    margin-bottom: 12px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.astra-card:hover {
    transform: translateY(-2px);
    border-color: rgba(124,58,237,0.45);
}
.astra-card h4 {
    margin: 0 0 6px 0;
    color: var(--astra-text);
    font-size: 1.05rem;
}
.astra-card p {
    margin: 0;
    color: var(--astra-muted);
    font-size: 0.9rem;
}

div[data-testid="stMetric"] {
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}
div[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--astra-muted) !important;
    font-weight: 500 !important;
}

div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: 0 !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(124,58,237,0.45) !important;
}
div.stButton > button:disabled {
    background: #2a324a !important;
    color: #64748b !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] {
    background: var(--astra-surface);
    border: 1px solid var(--astra-border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--astra-text) !important;
    font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 12px;
    overflow: hidden;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #2dd4bf) !important;
    border-radius: 6px;
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--astra-border) !important;
}

.astra-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: rgba(124,58,237,0.15);
    color: #c4b5fd;
    font-size: 0.78rem;
    border: 1px solid rgba(124,58,237,0.3);
    margin-right: 6px;
}
.astra-tag.ok {
    background: rgba(34,197,94,0.12);
    color: #86efac;
    border-color: rgba(34,197,94,0.3);
}
.astra-tag.warn {
    background: rgba(245,158,11,0.12);
    color: #fcd34d;
    border-color: rgba(245,158,11,0.3);
}

hr {
    border-color: var(--astra-border) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--astra-text) !important;
    letter-spacing: -0.01em;
}

.stRadio > div, .stSelectbox > div, .stTextArea textarea, .stNumberInput input {
    background: var(--astra-surface-2) !important;
    color: var(--astra-text) !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] section {
    background: var(--astra-surface) !important;
    border: 1px dashed rgba(148,163,184,0.35) !important;
    border-radius: 12px !important;
}

.astra-footer {
    margin-top: 28px;
    padding: 14px 18px;
    border-top: 1px solid var(--astra-border);
    color: var(--astra-muted);
    font-size: 0.8rem;
    text-align: center;
}
</style>
"""

STEP_LABELS = ["Data", "Pipeline", "Training", "Results"]


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="astra-hero">
            <div class="astra-hero-title">Astra ML System</div>
            <p class="astra-hero-sub">Automatic model selection driven by dataset profiling &mdash; configure, train, and ship.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(active: int) -> None:
    pieces = []
    for i, label in enumerate(STEP_LABELS):
        cls = "astra-step"
        if i < active:
            cls += " done"
        elif i == active:
            cls += " active"
        pieces.append(f'<div class="{cls}">{i + 1}. {label}</div>')
    st.markdown(f'<div class="astra-stepper">{"".join(pieces)}</div>', unsafe_allow_html=True)


def init_state() -> None:
    defaults: dict = {
        "step": 0,
        "data_path": None,
        "data_preview": None,
        "results": None,
        "profile": None,
        "best_model": None,
        "X_test": None,
        "y_test": None,
        "pipeline_config": None,
        "training_complete": False,
        "error_msg": None,
        "error_detail": None,
        "target_column": None,
        "feature_columns": [],
        "config_state": None,
        "model_comparison_history": None,
        "advanced_mode": False,
        "session_id": None,
        "output_dir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_DOMAIN_KEYS = (
    "step", "data_path", "data_preview", "results", "profile",
    "best_model", "X_test", "y_test", "pipeline_config",
    "training_complete", "error_msg", "error_detail",
    "target_column", "feature_columns", "config_state",
    "model_comparison_history", "advanced_mode",
    "session_id", "output_dir", "last_preset", "config_initialized",
    "preprocessing_summary", "target_classes",
    "adaptation_log", "best_result_meta",
)


def reset_session() -> None:
    for key in _DOMAIN_KEYS:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    init_state()


def init_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _seed_demo_datasets()


def _seed_demo_datasets() -> None:
    needed = {"iris.csv", "titanic.csv", "wine.csv"}
    existing = {p.name for p in DATA_DIR.glob("*.csv")}
    if needed.issubset(existing):
        return
    try:
        gen = DataGenerator(data_dir=DATA_DIR.parent)
        if "iris.csv" not in existing:
            gen.create_iris_dataset()
        if "titanic.csv" not in existing:
            gen.create_titanic_dataset()
        if "wine.csv" not in existing:
            gen.create_wine_dataset()
    except Exception as exc:
        logger.warning("Failed to seed demo datasets: %s", exc)


def setup_output_directory():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    existing = sorted(
        d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")
    )
    session_id = (int(existing[-1].name.split("_")[1]) + 1) if existing else 1
    out = OUTPUTS_DIR / f"run_{session_id:03d}"
    out.mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "visualizations").mkdir(exist_ok=True)
    return session_id, out


@st.cache_data(ttl=30)
def get_available_datasets() -> list[dict]:
    datasets = []
    if DATA_DIR.exists():
        for csv_file in DATA_DIR.glob("*.csv"):
            try:
                df_peek = pd.read_csv(csv_file, nrows=1)
                size_kb = csv_file.stat().st_size / 1024.0
                datasets.append(
                    {
                        "name": csv_file.stem,
                        "path": csv_file,
                        "cols": len(df_peek.columns),
                        "size_kb": size_kb,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping unreadable CSV %s: %s", csv_file, exc)
    return datasets


def _sanitize_filename(raw_name: str) -> str:
    stem = Path(raw_name).stem
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).strip("_") or "dataset"
    return safe[:64] + ".csv"


def _safe_unique_path(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return base_dir / f"{stem}_{stamp}{suffix}"


def detect_encoding(file_obj) -> str:
    if isinstance(file_obj, (str, Path)):
        with open(file_obj, "rb") as f:
            raw = f.read(10_000)
    else:
        raw = file_obj.read(10_000)
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    result = chardet.detect(raw)
    confidence = result.get("confidence") or 0
    encoding = result.get("encoding") or "utf-8"
    return encoding if confidence > 0.7 else "utf-8"


def auto_format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    seen: dict = {}
    new_cols = []
    for col in df_clean.columns:
        clean = re.sub(r"[^\w]", "_", str(col).strip(), flags=re.UNICODE).strip("_")
        clean = clean or f"unnamed_{len(new_cols)}"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df_clean.columns = new_cols

    df_clean.dropna(how="all", axis=0, inplace=True)
    df_clean.dropna(how="all", axis=1, inplace=True)

    for col in df_clean.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        if converted.notna().mean() >= 0.90:
            df_clean[col] = converted

    df_clean.replace(
        ["", " ", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"],
        np.nan,
        inplace=True,
    )
    df_clean.dropna(how="all", inplace=True)

    if len(df_clean.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns after cleaning.")
    return df_clean


def _read_csv_streaming(uploaded_file, encoding: str, sep: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        uploaded_file.seek(0)
        chunks: list[pd.DataFrame] = []
        rows_so_far = 0
        for chunk in pd.read_csv(uploaded_file, encoding=encoding, sep=sep, chunksize=20_000):
            rows_so_far += len(chunk)
            if rows_so_far > MAX_ROWS:
                return None, f"Dataset exceeds the maximum allowed rows ({MAX_ROWS:,})."
            chunks.append(chunk)
        if not chunks:
            return None, "Empty dataset."
        return pd.concat(chunks, ignore_index=True), None
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return None, f"Parse error: {type(exc).__name__}"


def format_and_save_csv(uploaded_file, save_path: Path) -> tuple[bool, str]:
    try:
        uploaded_file.seek(0, 2)
        size_mb = uploaded_file.tell() / (1024 ** 2)
        uploaded_file.seek(0)
        if size_mb > MAX_UPLOAD_MB:
            return False, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_MB} MB."

        encoding = detect_encoding(uploaded_file)
        uploaded_file.seek(0)

        df_head: Optional[pd.DataFrame] = None
        sep_used = ","
        for sep in (",", ";", "\t", "|"):
            try:
                uploaded_file.seek(0)
                tmp = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, nrows=100)
                if len(tmp.columns) > 1:
                    df_head = tmp
                    sep_used = sep
                    break
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
                continue

        if df_head is None or len(df_head.columns) <= 1:
            return False, "Could not parse CSV. Please verify the file format."

        df_full, err = _read_csv_streaming(uploaded_file, encoding, sep_used)
        if err is not None:
            return False, err
        if df_full is None:
            return False, "Could not read CSV content."

        df_fmt = auto_format_dataframe(df_full)

        if len(df_fmt) < 10:
            return False, f"Too few rows ({len(df_fmt)}). Minimum 10 recommended."

        df_fmt.to_csv(save_path, index=False, encoding="utf-8")
        return True, f"Saved (separator='{sep_used}', encoding={encoding})."

    except Exception:
        logger.exception("Error processing uploaded CSV")
        return False, "Could not process the file. See server logs for details."


def _preprocess_dataset(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, dict, Optional[list]]:
    summary: dict = {
        "rows_in": int(len(data)),
        "imputed_numeric": [],
        "imputed_categorical": [],
        "one_hot_columns": [],
        "label_encoded_columns": [],
        "dropped_high_cardinality": [],
        "target_encoded": False,
        "target_classes": None,
    }

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    work = data.copy()
    work = work.dropna(subset=[target_column])

    y_raw = work[target_column]
    target_classes: Optional[list] = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw.astype(str))
        target_classes = [str(c) for c in le.classes_]
        y_series = pd.Series(y_encoded, index=work.index, name=target_column)
        summary["target_encoded"] = True
        summary["target_classes"] = target_classes
    else:
        y_series = y_raw.copy()

    features = work.drop(columns=[target_column])

    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    object_cols = [c for c in features.columns if c not in numeric_cols]

    if numeric_cols:
        nan_cols = [c for c in numeric_cols if features[c].isna().any()]
        if nan_cols:
            imp = SimpleImputer(strategy="median")
            features[numeric_cols] = imp.fit_transform(features[numeric_cols])
            summary["imputed_numeric"] = nan_cols

    if object_cols:
        nan_cols = [c for c in object_cols if features[c].isna().any()]
        if nan_cols:
            for c in nan_cols:
                mode = features[c].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "unknown"
                features[c] = features[c].fillna(fill)
            summary["imputed_categorical"] = nan_cols

        low_card_cols: list[str] = []
        high_card_cols: list[str] = []
        very_high_card_cols: list[str] = []
        for c in object_cols:
            n_unique = features[c].nunique()
            if n_unique <= 1:
                very_high_card_cols.append(c)
            elif n_unique <= 10:
                low_card_cols.append(c)
            elif n_unique <= 50:
                high_card_cols.append(c)
            else:
                very_high_card_cols.append(c)

        if very_high_card_cols:
            features = features.drop(columns=very_high_card_cols)
            summary["dropped_high_cardinality"] = very_high_card_cols

        if low_card_cols:
            features = pd.get_dummies(
                features, columns=low_card_cols, drop_first=False, dtype=float
            )
            summary["one_hot_columns"] = low_card_cols

        for c in high_card_cols:
            le = LabelEncoder()
            features[c] = le.fit_transform(features[c].astype(str))
            summary["label_encoded_columns"].append(c)

    features = features.loc[:, features.apply(pd.api.types.is_numeric_dtype)]
    features = features.astype(float)
    y_series = y_series.loc[features.index]

    summary["rows_out"] = int(len(features))
    summary["feature_count_out"] = int(features.shape[1])
    return features, y_series, summary, target_classes


def set_data_and_advance(path: Path) -> None:
    try:
        st.session_state.data_path = path
        st.session_state.data_preview = pd.read_csv(path, nrows=100)
        st.session_state.step = 1
    except Exception:
        st.session_state.error_msg = f"Could not read '{path.name}'."
        logger.exception("Failed to read dataset: %s", path)


def load_preset_config(preset: str) -> None:
    cfg_map = {"Fast": get_fast_config, "Accurate": get_accurate_config}
    cfg = cfg_map.get(preset, get_default_config)()
    st.session_state.config_state = {
        "training": cfg.training.copy(),
        "scoring": cfg.scoring.copy(),
        "models": {
            k: {
                "name": v.name,
                "enabled": v.enabled,
                "params": v.params,
                "profile_requirements": v.profile_requirements,
                "description": v.description,
                "custom_scoring_weights": v.custom_scoring_weights or {
                    "accuracy_weight": cfg.scoring["accuracy_weight"],
                    "f1_weight": cfg.scoring["f1_weight"],
                    "cv_weight": cfg.scoring["cv_weight"],
                },
            }
            for k, v in cfg.models.items()
        },
    }


def build_config_from_state() -> PipelineConfig:
    cfg_data = st.session_state.config_state
    models: dict = {}

    for k, v in cfg_data["models"].items():
        try:
            raw_params = (
                v["params"] if isinstance(v["params"], dict) else json.loads(v["params"])
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON params for model '{k}': {exc}") from exc

        safe_params = validate_model_params(v["name"], raw_params)

        try:
            weights = (
                v["custom_scoring_weights"]
                if isinstance(v["custom_scoring_weights"], dict)
                else json.loads(v["custom_scoring_weights"])
            )
        except json.JSONDecodeError:
            weights = {}

        models[k] = ModelConfig(
            name=v["name"],
            enabled=v["enabled"],
            params=safe_params,
            profile_requirements=v["profile_requirements"],
            description=v["description"],
            custom_scoring_weights=weights,
        )

    return PipelineConfig(
        models=models,
        training=cfg_data["training"],
        scoring=cfg_data["scoring"],
    )


def save_visualizations(output_dir: Path, profile, best_model, X_test, y_test) -> None:
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    if profile:
        try:
            profile.visualize_profile(save_path=str(viz_dir / "data_profile.png"))
        except Exception as exc:
            logger.warning("Could not save profile visualisation: %s", exc)

    if best_model and best_model.is_trained:
        try:
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test),
            )
            ax.set_title(f"Confusion Matrix - {best_model.name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.savefig(str(viz_dir / "confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not save confusion matrix: %s", exc)


def save_results_to_session(
    output_dir: Path, session_id: int, results: dict,
    profile, best_model, selector, X_test, y_test,
) -> None:
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if profile:
        try:
            profile.save(str(output_dir / "profile.json"))
        except Exception as exc:
            logger.warning("Could not save profile: %s", exc)

    if selector:
        try:
            selector.save_all_models(str(output_dir / "models"))
        except Exception as exc:
            logger.warning("Could not save models: %s", exc)

    save_visualizations(output_dir, profile, best_model, X_test, y_test)


def render_artifacts_browser() -> None:
    st.divider()
    st.subheader("Artifacts & Outputs")
    if not OUTPUTS_DIR.exists():
        st.info("No artifacts yet. Run an analysis first.")
        return

    run_dirs = sorted(d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not run_dirs:
        st.info("No run directories found.")
        return

    run_path = run_dirs[-1].resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    st.info(f"Showing artifacts from **{run_path.name}**")

    files = sorted(f for f in run_path.rglob("*") if f.is_file())
    if not files:
        st.warning("No files in this directory.")
        return

    for file_path in files:
        resolved = file_path.resolve()
        try:
            resolved.relative_to(outputs_root)
        except ValueError:
            continue
        rel = resolved.relative_to(run_path)
        with st.expander(f"{rel}", expanded=False):
            col_view, col_dl = st.columns([1, 1])
            with col_view:
                suf = resolved.suffix.lower()
                if suf in (".png", ".jpg", ".jpeg", ".svg"):
                    st.image(str(resolved))
                elif suf == ".json":
                    try:
                        with open(resolved, "r", encoding="utf-8") as f:
                            st.json(json.load(f))
                    except Exception:
                        st.text("Could not render JSON.")
                elif suf == ".csv":
                    st.dataframe(pd.read_csv(resolved), use_container_width=True)
                elif suf == ".pkl":
                    st.text("Binary model file (download to inspect locally).")
                else:
                    st.text("No preview for this file type.")
            with col_dl:
                with open(resolved, "rb") as f:
                    key_hash = hashlib.sha1(
                        f"{run_path.name}|{rel}".encode("utf-8")
                    ).hexdigest()[:12]
                    st.download_button(
                        "Download",
                        data=f,
                        file_name=resolved.name,
                        mime="application/octet-stream",
                        key=f"dl_{key_hash}",
                    )


def render_comparison_chart(history: list, task_type: str = "classification") -> None:
    if not history:
        return

    if task_type == "classification":
        metric_keys = {"Accuracy": "accuracy", "F1-Score": "f1"}
    else:
        metric_keys = {"R2 Score": "r2", "MSE": "mse"}

    rows = []
    for entry in history:
        for label, key in metric_keys.items():
            val = float(entry["metrics"].get(key, 0.0) or 0.0)
            rows.append({
                "Model": entry["model"].replace("_Specialist", ""),
                "Metric": label,
                "Value": val,
                "Label": f"{val:.3f}",
            })

    df = pd.DataFrame(rows)
    model_order = (
        df.groupby("Model")["Value"].max().sort_values(ascending=False).index.tolist()
    )

    fig = px.bar(
        df,
        x="Model", y="Value", color="Metric",
        barmode="group",
        title="Model performance comparison",
        text="Label",
        color_discrete_sequence=["#a78bfa", "#34d399"],
        category_orders={"Model": model_order},
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color="#e6e9f2", size=11),
        cliponaxis=False,
    )
    fig.update_layout(
        yaxis_range=[0, 1.08] if task_type == "classification" else None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6e9f2",
        title_font_color="#e6e9f2",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, color="#94a3b8", type="category")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)", color="#94a3b8")
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_plotly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_labels: Optional[list] = None,
) -> None:
    try:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if class_labels is not None and len(class_labels) >= int(np.max(labels)) + 1:
            label_strs = [str(class_labels[int(l)]) for l in labels]
        else:
            label_strs = [str(l) for l in labels]
        text = [[str(int(v)) for v in row] for row in cm]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=label_strs,
                y=label_strs,
                colorscale=[
                    [0.0, "#0f172a"],
                    [0.25, "#3730a3"],
                    [0.55, "#7c3aed"],
                    [0.85, "#06b6d4"],
                    [1.0, "#5eead4"],
                ],
                text=text,
                texttemplate="%{text}",
                textfont=dict(color="#ffffff", size=14, family="Inter, sans-serif"),
                xgap=2,
                ygap=2,
                hovertemplate="True %{y} - Predicted %{x}: %{z}<extra></extra>",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Count", font=dict(color="#e6e9f2")),
                    tickfont=dict(color="#94a3b8"),
                    outlinewidth=0,
                ),
                zmin=0,
            )
        )
        n = max(len(label_strs), 1)
        height = max(360, min(640, 120 + 60 * n))
        fig.update_layout(
            title=dict(text=f"Confusion matrix - {model_name}", font=dict(color="#e6e9f2", size=18)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e6e9f2",
            xaxis=dict(
                title="Predicted",
                type="category",
                color="#94a3b8",
                showgrid=False,
                tickfont=dict(color="#cbd5e1"),
            ),
            yaxis=dict(
                title="True",
                type="category",
                color="#94a3b8",
                showgrid=False,
                tickfont=dict(color="#cbd5e1"),
                autorange="reversed",
            ),
            margin=dict(l=60, r=20, t=60, b=50),
            height=height,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        logger.warning("Confusion plot failed: %s", exc)
        st.info("Confusion matrix not available for this run.")


def step_dataset() -> None:
    st.header("Step 1 - Select data")

    datasets = get_available_datasets()
    if datasets:
        st.subheader("Available datasets")
        cols = st.columns(min(3, len(datasets)))
        for idx, ds in enumerate(datasets):
            with cols[idx % len(cols)]:
                st.markdown(
                    f"""
                    <div class="astra-card">
                        <h4>{ds['name']}</h4>
                        <p>{ds['cols']} columns &nbsp;·&nbsp; {ds['size_kb']:.1f} KB</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Use this dataset",
                    key=f"load_{ds['name']}",
                    on_click=set_data_and_advance,
                    args=(ds["path"],),
                    use_container_width=True,
                )
    else:
        st.info("No datasets found in `data/tabular`. Upload a CSV or create a test dataset below.")

    st.divider()
    col_upload, col_create = st.columns(2)

    with col_upload:
        with st.expander("Upload CSV (auto-format enabled)", expanded=False):
            st.markdown(
                """
**Requirements**
- First row = column headers
- At least one target column
- Minimum 10 rows
- Max file size 50 MB / 500,000 rows

**Auto-formatting**
- Encoding detection (UTF-8, Windows-1251, Latin-1)
- Separator detection (`,` `;` `\\t` `|`)
- Column name sanitisation
- Empty row/column removal
- Numeric string conversion
- Null normalisation
                """
            )
            uploaded = st.file_uploader("Select CSV", type=["csv"], key="upload_csv")
            if uploaded is not None:
                safe_name = _sanitize_filename(uploaded.name)
                save_path = _safe_unique_path(DATA_DIR, safe_name)

                with st.spinner("Auto-formatting..."):
                    success, message = format_and_save_csv(uploaded, save_path)
                if success:
                    st.success(message)
                    st.cache_data.clear()
                    try:
                        st.dataframe(pd.read_csv(save_path, nrows=10), use_container_width=True)
                        set_data_and_advance(save_path)
                        st.rerun()
                    except Exception:
                        logger.exception("Failed to load freshly saved CSV")
                        st.error("Could not read the freshly saved file.")
                else:
                    st.error(message)

    with col_create:
        with st.expander("Create test dataset", expanded=False):
            st.markdown("Generate built-in datasets for quick testing.")
            gen = DataGenerator(data_dir=Path("data"))
            creators = {
                "Iris":      gen.create_iris_dataset,
                "Wine":      gen.create_wine_dataset,
                "Digits":    gen.create_digits_dataset,
                "Titanic":   gen.create_titanic_dataset,
                "Synthetic": gen.create_synthetic_dataset,
                "Large":     gen.create_large_dataset,
            }
            for ds_name, creator in creators.items():
                if st.button(f"Create '{ds_name}'", key=f"create_{ds_name}", use_container_width=True):
                    with st.spinner(f"Creating '{ds_name}'..."):
                        try:
                            path = creator()
                            st.cache_data.clear()
                            set_data_and_advance(path)
                            st.rerun()
                        except Exception:
                            logger.exception("Dataset creation failed")
                            st.error("Failed to create dataset.")


def step_pipeline() -> None:
    st.header("Step 2 - Pipeline configuration")

    if st.session_state.data_preview is None:
        st.warning("Please select data in Step 1 first.")
        if st.button("Back to data selection"):
            st.session_state.step = 0
            st.rerun()
        return

    st.markdown(
        f'<span class="astra-tag ok">Dataset: {st.session_state.data_path.name}</span>',
        unsafe_allow_html=True,
    )
    st.dataframe(st.session_state.data_preview, use_container_width=True)

    cols = list(st.session_state.data_preview.columns)
    _candidates = ["target", "class", "label", "category", "target_name"]
    default_idx = next(
        (cols.index(c) for c in _candidates if c in cols),
        len(cols) - 1,
    )
    target_col = st.selectbox(
        "Target column",
        options=cols,
        index=default_idx,
        help="Select the column the model should learn to predict.",
    )
    st.session_state.target_column = target_col

    st.divider()

    if st.session_state.config_state is None:
        load_preset_config("Standard")

    preset_mode = st.radio(
        "Configuration mode",
        ["Fast", "Standard", "Accurate", "Advanced Customization"],
        index=1,
        key="preset_mode_radio",
        horizontal=True,
    )

    if preset_mode != "Advanced Customization":
        if st.session_state.get("last_preset") != preset_mode or st.session_state.config_state is None:
            load_preset_config(preset_mode)
            st.session_state.last_preset = preset_mode

        cfg = st.session_state.config_state
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Models enabled", sum(1 for m in cfg["models"].values() if m["enabled"]))
        c2.metric("CV folds", cfg["training"].get("cv_folds", 5))
        c3.metric("Test size", f"{cfg['training'].get('test_size', 0.2):.0%}")
        c4.metric("Mode", preset_mode)
    else:
        st.subheader("Advanced pipeline configuration")
        st.warning("Fine-tune all parameters manually.")

        if not st.session_state.get("config_initialized"):
            load_preset_config("Accurate")
            st.session_state.config_initialized = True

        with st.form("advanced_config_form"):
            cfg_s = st.session_state.config_state
            cv_folds = st.number_input(
                "CV folds", value=cfg_s["training"].get("cv_folds", 5), min_value=2, max_value=20,
            )
            test_size = st.slider(
                "Test size", 0.1, 0.4, value=cfg_s["training"].get("test_size", 0.2), step=0.05,
            )
            use_cv = st.checkbox(
                "Use CV in final scoring",
                value=cfg_s["training"].get("use_cv_in_scoring", False),
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                aw = st.slider(
                    "Global accuracy weight", 0.0, 1.0,
                    value=cfg_s["scoring"].get("accuracy_weight", 0.7), step=0.05,
                )
            with c2:
                fw = st.slider(
                    "Global F1 weight", 0.0, 1.0,
                    value=cfg_s["scoring"].get("f1_weight", 0.3), step=0.05,
                )
            with c3:
                cw = st.slider(
                    "Global CV weight", 0.0, 1.0,
                    value=cfg_s["scoring"].get("cv_weight", 0.0), step=0.05,
                )

            st.divider()
            st.markdown("### Model-specific configuration")

            model_updates: dict = {}
            for model_key, model_data in cfg_s["models"].items():
                with st.expander(f"{model_data['name']}", expanded=False):
                    col_en, col_desc = st.columns([1, 3])
                    with col_en:
                        enabled = st.checkbox(
                            "Enabled", value=model_data.get("enabled", True),
                            key=f"en_{model_key}",
                        )
                    with col_desc:
                        st.markdown(f"*{model_data.get('description', '')}*")

                    st.markdown("**Hyperparameters (JSON)**")
                    params_json = json.dumps(model_data.get("params", {}), indent=2)
                    raw_params_str = st.text_area(
                        "Params", value=params_json, height=100, key=f"p_{model_key}",
                    )

                    try:
                        parsed_params = json.loads(raw_params_str)
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid JSON: {exc}")
                        parsed_params = model_data.get("params", {})

                    st.markdown("**Custom scoring weights** *(overrides global)*")
                    wc1, wc2, wc3 = st.columns(3)
                    existing_w = model_data.get("custom_scoring_weights", {})
                    with wc1:
                        m_aw = st.slider(
                            "Acc", 0.0, 1.0,
                            value=existing_w.get("accuracy_weight", cfg_s["scoring"]["accuracy_weight"]),
                            step=0.05, key=f"aw_{model_key}",
                        )
                    with wc2:
                        m_fw = st.slider(
                            "F1", 0.0, 1.0,
                            value=existing_w.get("f1_weight", cfg_s["scoring"]["f1_weight"]),
                            step=0.05, key=f"fw_{model_key}",
                        )
                    with wc3:
                        m_cw = st.slider(
                            "CV", 0.0, 1.0,
                            value=existing_w.get("cv_weight", cfg_s["scoring"]["cv_weight"]),
                            step=0.05, key=f"cw_{model_key}",
                        )

                    model_updates[model_key] = {
                        "enabled": enabled,
                        "params": parsed_params,
                        "custom_scoring_weights": {
                            "accuracy_weight": m_aw,
                            "f1_weight": m_fw,
                            "cv_weight": m_cw,
                        },
                    }

            submitted = st.form_submit_button("Apply advanced configuration", type="primary")
            if submitted:
                cfg_s["training"]["cv_folds"] = cv_folds
                cfg_s["training"]["test_size"] = test_size
                cfg_s["training"]["use_cv_in_scoring"] = use_cv
                cfg_s["scoring"]["accuracy_weight"] = aw
                cfg_s["scoring"]["f1_weight"] = fw
                cfg_s["scoring"]["cv_weight"] = cw
                for mk, upd in model_updates.items():
                    cfg_s["models"][mk].update(upd)
                st.success("Advanced configuration applied.")

    col_back, col_run = st.columns(2)
    with col_back:
        if st.button("Back"):
            st.session_state.step = 0
            st.rerun()
    with col_run:
        if st.button("Save & start training", type="primary"):
            try:
                config_obj = build_config_from_state()
                st.session_state.pipeline_config = config_obj
                st.session_state.step = 2
                st.rerun()
            except ValueError as exc:
                st.error(f"Configuration error: {exc}")


def step_training() -> None:
    st.header("Step 3 - Model training")

    if st.session_state.data_path is None or st.session_state.pipeline_config is None:
        st.warning("Missing data or configuration. Please go back.")
        if st.button("Back to configuration"):
            st.session_state.step = 1
            st.rerun()
        return

    if st.session_state.training_complete:
        st.success("Training completed.")
        if st.button("View results", type="primary"):
            st.session_state.step = 3
            st.rerun()
        return

    session_id, output_dir = setup_output_directory()
    st.session_state.session_id = session_id
    st.session_state.output_dir = output_dir

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.markdown("Loading dataset...")
        progress_bar.progress(10)
        try:
            file_size_mb = st.session_state.data_path.stat().st_size / (1024 ** 2)
        except OSError:
            file_size_mb = 0.0
        if file_size_mb > MAX_UPLOAD_MB:
            raise ValueError(
                f"Dataset is too large ({file_size_mb:.1f} MB). "
                f"Max allowed: {MAX_UPLOAD_MB} MB."
            )
        data_chunks: list[pd.DataFrame] = []
        rows_loaded = 0
        for chunk in pd.read_csv(st.session_state.data_path, chunksize=20_000):
            rows_loaded += len(chunk)
            if rows_loaded > MAX_ROWS:
                raise ValueError(
                    f"Dataset exceeds the maximum allowed rows ({MAX_ROWS:,})."
                )
            data_chunks.append(chunk)
        if not data_chunks:
            raise ValueError("Dataset is empty.")
        data = pd.concat(data_chunks, ignore_index=True)
        config = st.session_state.pipeline_config

        target_column = st.session_state.get("target_column")
        if target_column is None or target_column not in data.columns:
            for candidate in ("target", "class", "label", "category", "target_name"):
                if candidate in data.columns:
                    target_column = candidate
                    break
            else:
                num_cols = data.select_dtypes(include=np.number).columns
                if len(num_cols) > 1:
                    target_column = num_cols[-1]
                else:
                    raise ValueError("Could not find a target column. Please go back to Step 2.")

        st.session_state.target_column = target_column

        status_text.markdown("Preprocessing data...")
        progress_bar.progress(15)
        X_df, y_series, prep_summary, target_classes = _preprocess_dataset(
            data, target_column
        )
        st.session_state.feature_columns = list(X_df.columns)
        st.session_state.preprocessing_summary = prep_summary
        st.session_state.target_classes = target_classes

        if X_df.shape[1] < 2:
            raise ValueError(
                "Need at least 2 feature columns after preprocessing. "
                "Dataset has too few usable columns."
            )
        if X_df.shape[0] < 30:
            raise ValueError(
                f"Dataset has only {X_df.shape[0]} rows after cleaning. "
                "Minimum 30 required for stable training."
            )
        if y_series.nunique() < 2:
            raise ValueError(
                "Target column has fewer than 2 unique values. "
                "Cannot train a classifier on a single class."
            )

        X = X_df.values
        y = y_series.values

        status_text.markdown("Profiling data...")
        progress_bar.progress(25)
        profiler = DataProfiler(dataset_name=st.session_state.data_path.stem)
        profile = profiler.profile_tabular_data(
            X, y, feature_names=list(X_df.columns)
        )
        st.session_state.profile = profile
        task_type = profile.task_type

        status_text.markdown("Preparing models with profile-aware adaptation...")
        progress_bar.progress(35)
        selector = AdaptiveModelSelector()
        selector.register_models_from_pipeline_config(
            config, task_type=task_type, data_profile=profile.to_dict()
        )
        st.session_state.adaptation_log = dict(
            getattr(selector, "adaptation_log", {})
        )
        status_text.markdown(
            f"Training {len(selector.models)} model(s) with profile-aware adaptation..."
        )

        def _on_model_start(idx: int, total: int, name: str) -> None:
            pct = 35 + int(50 * ((idx - 1) / max(total, 1)))
            try:
                progress_bar.progress(min(85, pct))
                status_text.markdown(
                    f"Training [{idx}/{total}]: **{name.replace('_Specialist','')}**"
                )
            except Exception:
                pass

        best_model, X_test, y_test = selector.profile_and_select(
            X, y,
            data_profile=profile.to_dict(),
            cv_folds=config.training.get("cv_folds", 5),
            test_size=config.training.get("test_size", 0.2),
            random_state=config.training.get("random_state", 42),
            use_cv_in_scoring=config.training.get("use_cv_in_scoring", False),
            global_accuracy_weight=config.scoring.get("accuracy_weight", 0.7),
            global_f1_weight=config.scoring.get("f1_weight", 0.3),
            global_cv_weight=config.scoring.get("cv_weight", 0.0),
            progress_callback=_on_model_start,
        )
        st.session_state.best_result_meta = getattr(selector, "best_result", None)

        st.session_state.best_model = best_model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        if hasattr(selector, "selection_history") and selector.selection_history:
            st.session_state.model_comparison_history = selector.selection_history

        metrics = best_model.performance_metrics
        if task_type == "classification":
            primary_metric = {
                "accuracy": float(metrics.get("accuracy", 0)),
                "f1": float(metrics.get("f1", 0)),
            }
        else:
            primary_metric = {
                "r2": float(metrics.get("r2", 0)),
                "rmse": float(metrics.get("rmse", 0)),
            }

        st.session_state.results = {
            "session_id": session_id,
            "task_type": task_type,
            "best_model": best_model.name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "timestamp": str(datetime.now()),
            **primary_metric,
        }

        progress_bar.progress(90)
        status_text.markdown("Saving artefacts...")
        save_results_to_session(
            output_dir, session_id, st.session_state.results,
            profile, best_model, selector, X_test, y_test,
        )
        progress_bar.progress(100)
        status_text.empty()

        st.session_state.training_complete = True
        st.session_state.step = 3
        st.rerun()

    except Exception as exc:
        logger.exception("Training failed")
        st.session_state.error_msg = f"Training error: {type(exc).__name__}"
        st.session_state.error_detail = traceback.format_exc()
        st.rerun()


def _render_profile_panel() -> None:
    profile = st.session_state.get("profile")
    if profile is None:
        return
    p = profile.to_dict()
    st.subheader("Working profile detected")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{p['n_samples']:,}")
    c2.metric("Features", p["n_features"])
    c3.metric("Complexity", p["data_complexity"].upper())
    if p.get("n_classes") is not None:
        c4.metric("Classes", p["n_classes"])
    else:
        c4.metric("Task", p.get("task_type", "regression").title())

    rec = p.get("recommended_models") or []
    needs = p.get("preprocessing_needs") or []
    if rec:
        rec_tags = " ".join(f'<span class="astra-tag">{m}</span>' for m in rec)
        st.markdown(
            f"**Profile-recommended families:** {rec_tags}",
            unsafe_allow_html=True,
        )
    if needs and needs != ["none_required"]:
        warn_tags = " ".join(
            f'<span class="astra-tag warn">{n.replace("_"," ")}</span>' for n in needs
        )
        st.markdown(
            f"**Profile-detected preprocessing needs:** {warn_tags}",
            unsafe_allow_html=True,
        )

    balance = p.get("class_balance_ratio")
    if balance is not None and balance < 0.5:
        st.warning(
            f"Class balance ratio is {balance:.2f} (<0.5) — "
            "imbalanced dataset, prefer balanced or boosted models."
        )

    with st.expander("Raw profile JSON", expanded=False):
        st.json({
            "n_samples": p["n_samples"],
            "n_features": p["n_features"],
            "n_classes": p.get("n_classes"),
            "task_type": p.get("task_type"),
            "data_complexity": p["data_complexity"],
            "class_balance_ratio": p.get("class_balance_ratio"),
            "recommended_models": rec,
            "preprocessing_needs": needs,
        })


def _render_preprocessing_panel() -> None:
    summary = st.session_state.get("preprocessing_summary")
    if not summary:
        return
    has_actions = any(
        summary.get(k)
        for k in (
            "imputed_numeric", "imputed_categorical",
            "one_hot_columns", "label_encoded_columns",
            "dropped_high_cardinality", "target_encoded",
        )
    )
    if not has_actions:
        return
    with st.expander("Preprocessing applied", expanded=False):
        lines = []
        if summary.get("rows_in") != summary.get("rows_out"):
            lines.append(
                f"- Rows: {summary['rows_in']:,} → {summary['rows_out']:,}"
            )
        if summary.get("imputed_numeric"):
            lines.append(
                f"- Imputed NaN (median) in numeric columns: "
                f"{', '.join(summary['imputed_numeric'])}"
            )
        if summary.get("imputed_categorical"):
            lines.append(
                f"- Imputed NaN (mode) in categorical columns: "
                f"{', '.join(summary['imputed_categorical'])}"
            )
        if summary.get("one_hot_columns"):
            lines.append(
                f"- One-hot encoded (≤10 unique): "
                f"{', '.join(summary['one_hot_columns'])}"
            )
        if summary.get("label_encoded_columns"):
            lines.append(
                f"- Label encoded (10–50 unique): "
                f"{', '.join(summary['label_encoded_columns'])}"
            )
        if summary.get("dropped_high_cardinality"):
            lines.append(
                f"- Dropped (>50 unique or constant): "
                f"{', '.join(summary['dropped_high_cardinality'])}"
            )
        if summary.get("target_encoded"):
            classes = summary.get("target_classes") or []
            preview = ", ".join(classes[:6]) + ("…" if len(classes) > 6 else "")
            lines.append(
                f"- Target label-encoded: {len(classes)} classes [{preview}]"
            )
        st.markdown("\n".join(lines))


def _render_selection_explanation() -> None:
    history = st.session_state.get("model_comparison_history") or []
    best_meta = st.session_state.get("best_result_meta")
    adaptation = st.session_state.get("adaptation_log") or {}
    if not history and not best_meta:
        return

    st.subheader("Why this model was selected")

    if best_meta:
        best_name = best_meta["model"].replace("_Specialist", "")
        profile_score = best_meta.get("profile_score", 0.0)
        metric_score = best_meta.get("metric_score", 0.0)
        final_score = best_meta.get("final_score", 0.0)
        st.markdown(
            f"**{best_name}** was chosen with a final score of **{final_score:.3f}**, "
            f"combining a profile-match of **{profile_score:.2f}** "
            f"and a metric score of **{metric_score:.3f}** "
            f"(weighted 75% metrics / 25% profile fit)."
        )

    if history:
        rivals = sorted(history, key=lambda e: e.get("final_score", 0), reverse=True)[1:3]
        if rivals and best_meta:
            best_final = best_meta.get("final_score", 0.0)
            for r in rivals:
                gap = best_final - r.get("final_score", 0.0)
                st.caption(
                    f"vs **{r['model'].replace('_Specialist','')}**: "
                    f"−{gap:.3f} ({r.get('metric_score',0):.3f} metric · "
                    f"{r.get('profile_score',0):.2f} profile)"
                )

    if adaptation:
        with st.expander("Profile-driven hyperparameter adaptation", expanded=False):
            for name, notes in adaptation.items():
                if not notes:
                    continue
                short = name.replace("_Specialist", "")
                st.markdown(f"**{short}**")
                for n in notes:
                    st.markdown(f"  · {n}")


def step_results() -> None:
    st.header("Step 4 - Results")

    if not st.session_state.training_complete or st.session_state.results is None:
        st.warning("No results available.")
        if st.button("Back to training"):
            st.session_state.step = 2
            st.rerun()
        return

    res = st.session_state.results
    task_type: str = res.get("task_type", "classification")

    if st.session_state.session_id:
        st.markdown(
            f'<span class="astra-tag ok">Session #{st.session_state.session_id:03d} completed</span>',
            unsafe_allow_html=True,
        )
        st.write("")

    if task_type == "classification":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best model", res["best_model"].replace("_Specialist", ""))
        with col2:
            st.metric("Accuracy", f"{res.get('accuracy', 0):.4f}")
        with col3:
            st.metric("F1 score", f"{res.get('f1', 0):.4f}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best model", res["best_model"].replace("_Specialist", ""))
        with col2:
            st.metric("R2 score", f"{res.get('r2', 0):.4f}")
        with col3:
            st.metric("RMSE", f"{res.get('rmse', 0):.4f}")

    _render_profile_panel()
    _render_preprocessing_panel()
    _render_selection_explanation()

    if (
        task_type == "classification"
        and st.session_state.best_model is not None
        and st.session_state.X_test is not None
        and st.session_state.y_test is not None
        and st.session_state.best_model.is_trained
    ):
        try:
            y_pred = st.session_state.best_model.predict(st.session_state.X_test)
            render_confusion_plotly(
                st.session_state.y_test, y_pred,
                st.session_state.best_model.name.replace("_Specialist", ""),
                class_labels=st.session_state.get("target_classes"),
            )
        except Exception as exc:
            logger.warning("Could not render confusion matrix: %s", exc)

    if st.session_state.model_comparison_history:
        history = st.session_state.model_comparison_history

        with st.expander("Detailed comparison table", expanded=False):
            rows = []
            for entry in history:
                row = {
                    "Model": entry["model"].replace("_Specialist", ""),
                    "Profile score": f"{entry['profile_score']:.2f}",
                    "Final score": f"{entry['final_score']:.4f}",
                }
                for k, v in entry["metrics"].items():
                    row[k.upper()] = f"{v:.4f}"
                if entry.get("cv_results") and entry["cv_results"].get("cv_mean"):
                    row["CV mean"] = f"{entry['cv_results']['cv_mean']:.4f}"
                rows.append(row)

            df_comp = pd.DataFrame(rows)
            df_comp = df_comp.sort_values("Final score", ascending=False).reset_index(drop=True)
            st.dataframe(df_comp, use_container_width=True)

        render_comparison_chart(history, task_type)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        report_json = json.dumps(res, indent=2, ensure_ascii=False)
        st.download_button(
            "Download report (JSON)",
            data=report_json,
            file_name="astra_report.json",
            mime="application/json",
        )
    with col_dl2:
        if st.session_state.best_model and st.session_state.best_model.is_trained:
            import joblib
            buf = io.BytesIO()
            joblib.dump(st.session_state.best_model.model, buf)
            st.download_button(
                "Download best model (.pkl)",
                data=buf.getvalue(),
                file_name=f"{st.session_state.best_model.name}.pkl",
                mime="application/octet-stream",
            )

    render_artifacts_browser()

    if st.session_state.output_dir:
        st.success(f"All artifacts saved to: {st.session_state.output_dir}")

    if st.button("Start new analysis", type="primary"):
        reset_session()
        st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Astra ML")
        st.caption(f"v{APP_VERSION} · dataset-driven model selection")
        st.divider()

        step = st.session_state.get("step", 0)
        st.markdown(f"**Current step:** {STEP_LABELS[step]}")
        st.progress((step + 1) / len(STEP_LABELS))

        if st.session_state.get("data_path"):
            st.markdown("**Dataset**")
            st.caption(st.session_state.data_path.name)

        if st.session_state.get("session_id"):
            st.markdown("**Session**")
            st.caption(f"run_{st.session_state.session_id:03d}")

        st.divider()
        st.markdown("**Quick actions**")
        if st.button("Reset session", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        st.caption(
            "Models: RandomForest, SVM, Gradient Boosting, Neural Net, "
            "Logistic Regression."
        )


def main() -> None:
    init_state()
    init_directories()
    inject_css()

    render_sidebar()
    render_hero()
    render_stepper(st.session_state.step)

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        if SHOW_TRACEBACK_IN_UI and st.session_state.get("error_detail"):
            with st.expander("Technical details", expanded=False):
                st.code(st.session_state.error_detail)
        if st.button("Clear error and restart"):
            reset_session()
            st.rerun()
        return

    step = st.session_state.step
    if step == 0:
        step_dataset()
    elif step == 1:
        step_pipeline()
    elif step == 2:
        step_training()
    elif step == 3:
        step_results()

    st.markdown(
        f'<div class="astra-footer">'
        f'Astra ML &middot; v{APP_VERSION} &middot; {APP_YEAR} &middot; {APP_AUTHOR}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
