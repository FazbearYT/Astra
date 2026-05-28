"""
Astra ML System — Streamlit web application.

UI
--
Dark-themed, sidebar-driven, 4-step wizard
(Data -> Pipeline -> Training -> Results).

Security hardening applied
--------------------------
* Path traversal (CWE-22): uploaded filenames are sanitised before being used
  as filesystem paths.
* DoS via large uploads: Streamlit is configured to reject files > 50 MB
  (see .streamlit/config.toml); the app additionally enforces MAX_ROWS.
* Arbitrary model params leading to resource exhaustion: params are validated
  by `validate_model_params()` from model_selector before use.
* Pickle deserialization (CWE-502): joblib.load is only called for files
  written by the app itself inside the `outputs/` directory.  User-supplied
  binary files are never loaded.
* Full stack traces are written to logs only — users see only a short message.
"""
from __future__ import annotations

import io
import json
import logging
import os
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
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/tabular")
OUTPUTS_DIR = Path("outputs")
MAX_ROWS = 500_000          # hard limit for uploaded datasets
MAX_COLS = 1_000            # CWE-400: reject wide CSVs (CSV-bomb mitigation)
MAX_UPLOAD_MB = 50          # soft limit (also enforced in config.toml)

# CWE-209: full tracebacks leak file paths and internals. Only show them when
# ASTRA_DEBUG=1 is set in the environment.
DEBUG_MODE = os.environ.get("ASTRA_DEBUG", "0").lower() in ("1", "true", "yes")

# Trusted root for the artefacts browser. Anything resolved outside this is
# rejected (CWE-22: symlinks inside outputs/ could otherwise leak files).
_OUTPUTS_ROOT = OUTPUTS_DIR.resolve()

st.set_page_config(
    page_title="Astra ML",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# UI theme — custom CSS + horizontal stepper
# ---------------------------------------------------------------------------

_STEP_LABELS = ("Data", "Pipeline", "Training", "Results")

_CUSTOM_CSS = """
<style>
:root {
    --astra-bg: #0E1117;
    --astra-surface: #161B26;
    --astra-surface-2: #1F2533;
    --astra-border: #2A3142;
    --astra-text: #E6E8EE;
    --astra-muted: #8A93A6;
    --astra-accent: #7C5CFF;
    --astra-accent-dim: #5C45B8;
    --astra-success: #4ADE80;
    --astra-danger: #F87171;
}

.block-container { padding-top: 1.5rem; max-width: 1180px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: var(--astra-surface);
    color: var(--astra-text);
    border: 1px solid var(--astra-border);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: background 0.15s, border-color 0.15s, transform 0.05s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--astra-accent);
    background: var(--astra-surface-2);
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"] {
    background: var(--astra-accent);
    border-color: var(--astra-accent);
    color: #fff;
}
.stButton > button[kind="primary"]:hover {
    background: var(--astra-accent-dim);
    border-color: var(--astra-accent-dim);
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
    background: var(--astra-surface) !important;
    border: 1px solid var(--astra-border) !important;
    color: var(--astra-text) !important;
    border-radius: 8px !important;
}

/* Expanders + tabs + metrics */
div[data-testid="stExpander"] {
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 12px;
    overflow: hidden;
}
div[data-testid="stMetric"] {
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 12px;
    padding: 14px 16px;
}

/* DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--astra-border);
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
div[data-testid="stAlert"] { border-radius: 10px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B0E14;
    border-right: 1px solid var(--astra-border);
}
section[data-testid="stSidebar"] .stRadio label { color: var(--astra-text); }

/* Brand header */
.astra-brand {
    display: flex; align-items: center; gap: 14px;
    padding: 4px 0 18px 0;
}
.astra-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #7C5CFF 0%, #38BDF8 100%);
    display: flex; align-items: center; justify-content: center;
    color: #0E1117; font-weight: 700; font-size: 18px;
}
.astra-title { font-size: 24px; font-weight: 600; margin: 0; }
.astra-sub  { color: var(--astra-muted); font-size: 13px; margin: 2px 0 0 0; }

/* Stepper */
.astra-stepper {
    display: flex; justify-content: space-between;
    background: var(--astra-surface);
    border: 1px solid var(--astra-border);
    border-radius: 14px;
    padding: 14px 22px;
    margin: 6px 0 22px 0;
}
.astra-step { display: flex; align-items: center; gap: 10px; flex: 1; }
.astra-step + .astra-step::before {
    content: ""; flex: 0 0 28px; height: 2px;
    background: var(--astra-border); margin-right: 10px;
}
.astra-step.done + .astra-step::before { background: var(--astra-accent); }
.astra-bullet {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600;
    background: var(--astra-surface-2);
    color: var(--astra-muted);
    border: 1px solid var(--astra-border);
}
.astra-step.done .astra-bullet {
    background: var(--astra-accent); color: #fff; border-color: var(--astra-accent);
}
.astra-step.active .astra-bullet {
    background: #0E1117; color: var(--astra-accent); border-color: var(--astra-accent);
    box-shadow: 0 0 0 4px rgba(124, 92, 255, 0.15);
}
.astra-step-label { color: var(--astra-muted); font-size: 13px; font-weight: 500; }
.astra-step.active .astra-step-label { color: var(--astra-text); }
.astra-step.done   .astra-step-label { color: var(--astra-text); }

/* Section header */
.astra-section {
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 6px 0 14px 0;
}
.astra-section h2 { margin: 0; font-size: 22px; font-weight: 600; }
.astra-section .hint { color: var(--astra-muted); font-size: 13px; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_brand() -> None:
    st.markdown(
        """
        <div class="astra-brand">
            <div class="astra-logo">A</div>
            <div>
                <div class="astra-title">Astra ML</div>
                <div class="astra-sub">Automatic model selection based on dataset profiling</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current: int) -> None:
    """Render a horizontal 4-step indicator. ``current`` is 0-indexed."""
    parts = []
    for i, label in enumerate(_STEP_LABELS):
        cls = "astra-step"
        if i < current:
            cls += " done"
        elif i == current:
            cls += " active"
        parts.append(
            f'<div class="{cls}">'
            f'  <div class="astra-bullet">{i + 1}</div>'
            f'  <div class="astra-step-label">{label}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="astra-stepper">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="astra-section"><h2>{title}</h2>'
        f'<span class="hint">{hint}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

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
        "error_detail": None,      # full traceback, never shown to UI
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


def reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


# ---------------------------------------------------------------------------
# Directory / session helpers
# ---------------------------------------------------------------------------

def init_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


# ---------------------------------------------------------------------------
# Dataset listing — cached to avoid re-reading headers on every render
# FIX: @st.cache_data with TTL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_available_datasets() -> list[dict]:
    datasets = []
    if DATA_DIR.exists():
        for csv_file in DATA_DIR.glob("*.csv"):
            try:
                df_peek = pd.read_csv(csv_file, nrows=1)
                datasets.append(
                    {"name": csv_file.stem, "path": csv_file, "cols": len(df_peek.columns)}
                )
            except Exception as exc:
                logger.warning("Skipping unreadable CSV %s: %s", csv_file, exc)
    return datasets


# ---------------------------------------------------------------------------
# File-upload helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(raw_name: str) -> str:
    """FIX (CWE-22): strip path components and dangerous characters.

    Returns a safe filename with .csv extension, max 64 characters.
    """
    stem = Path(raw_name).stem
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem).strip("_") or "dataset"
    return safe[:64] + ".csv"


def detect_encoding(file_obj) -> str:
    """Detect CSV encoding using chardet."""
    if isinstance(file_obj, (str, Path)):
        with open(file_obj, "rb") as f:
            raw = f.read(10_000)
    else:
        raw = file_obj.read(10_000)
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    result = chardet.detect(raw)
    return result["encoding"] if (result["confidence"] or 0) > 0.7 else "utf-8"


def auto_format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw DataFrame for ML pipeline compatibility."""
    df_clean = df.copy()

    # Normalise column names
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

    # Drop fully empty rows / columns
    df_clean.dropna(how="all", axis=0, inplace=True)
    df_clean.dropna(how="all", axis=1, inplace=True)

    # FIX: use errors='coerce' so mixed columns convert properly;
    # only apply conversion when at least 90 % of values parse successfully.
    # Log conversions so the user knows what was silently changed.
    converted_cols: list[str] = []
    for col in df_clean.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        if converted.notna().mean() >= 0.90:
            df_clean[col] = converted
            converted_cols.append(col)
    if converted_cols:
        logger.info("Auto-converted object → numeric: %s", converted_cols)

    # Normalise common null representations
    df_clean.replace(["", " ", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"], np.nan, inplace=True)

    df_clean.dropna(how="all", inplace=True)

    if len(df_clean.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns after cleaning.")
    return df_clean


def format_and_save_csv(uploaded_file, save_path: Path) -> tuple[bool, str]:
    """Parse, clean and save an uploaded CSV file.

    FIX (CWE-22): save_path must already be sanitised by the caller.
    """
    try:
        # Check raw upload size before parsing
        uploaded_file.seek(0, 2)
        size_mb = uploaded_file.tell() / (1024 ** 2)
        uploaded_file.seek(0)
        if size_mb > MAX_UPLOAD_MB:
            return False, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_MB} MB."

        encoding = detect_encoding(uploaded_file)
        uploaded_file.seek(0)

        df: Optional[pd.DataFrame] = None
        sep_used = ","
        for sep in (",", ";", "\t", "|"):
            try:
                uploaded_file.seek(0)
                tmp = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, nrows=100)
                if len(tmp.columns) > 1:
                    df = tmp
                    sep_used = sep
                    break
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
                # Try next separator — failure here is expected and handled below
                continue

        if df is None or len(df.columns) <= 1:
            return False, "Could not parse CSV. Please verify the file format."

        # CWE-400: reject pathologically wide CSVs before pandas materialises
        # the full DataFrame. We already know column count from the sniff above.
        if len(df.columns) > MAX_COLS:
            return False, (
                f"CSV has {len(df.columns):,} columns. Maximum allowed: {MAX_COLS:,}."
            )

        uploaded_file.seek(0)
        df_full = pd.read_csv(uploaded_file, encoding=encoding, sep=sep_used)

        if len(df_full) > MAX_ROWS:
            return False, (
                f"Dataset has {len(df_full):,} rows. Maximum allowed: {MAX_ROWS:,}."
            )
        if len(df_full.columns) > MAX_COLS:
            return False, (
                f"CSV has {len(df_full.columns):,} columns. Maximum allowed: {MAX_COLS:,}."
            )

        df_fmt = auto_format_dataframe(df_full)

        if len(df_fmt) < 10:
            return False, f"Too few rows ({len(df_fmt)}). Minimum 10 recommended."

        df_fmt.to_csv(save_path, index=False, encoding="utf-8")
        return True, f"Saved successfully (separator='{sep_used}', encoding={encoding})."

    except Exception as exc:
        logger.exception("Error processing uploaded CSV")
        return False, f"Error processing file: {exc}"


def set_data_and_advance(path: Path) -> None:
    try:
        st.session_state.data_path = path
        st.session_state.data_preview = pd.read_csv(path, nrows=100)
        st.session_state.step = 1
    except Exception:  # noqa: BLE001
        st.session_state.error_msg = f"Could not read '{path.name}'."
        logger.exception("Failed to read dataset: %s", path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

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
    """Build PipelineConfig from session_state, validating all model params.

    FIX: validate params via validate_model_params() to prevent DoS.
    FIX: catch specific exceptions instead of bare except.
    """
    cfg_data = st.session_state.config_state
    models: dict = {}

    for k, v in cfg_data["models"].items():
        # Parse params JSON if string
        try:
            raw_params = (
                v["params"] if isinstance(v["params"], dict) else json.loads(v["params"])
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON params for model '{k}': {exc}") from exc

        # Validate / sanitise params
        safe_params = validate_model_params(v["name"], raw_params)

        # Parse weights JSON if string
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


# ---------------------------------------------------------------------------
# Output / artefacts helpers
# ---------------------------------------------------------------------------

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
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            ax.set_title(f"Confusion Matrix — {best_model.name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            fig.savefig(str(viz_dir / "confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not save confusion matrix: %s", exc)


def save_results_to_session(
    output_dir: Path, session_id: int, results: dict,
    profile, best_model, selector, X_test, y_test
) -> None:
    # results.json
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


def _is_inside_outputs(path: Path) -> bool:
    """CWE-22 guard: ensure *path* resolves inside the trusted outputs root.

    Rejects symlinks that escape OUTPUTS_DIR (e.g. ``outputs/run_001/leak ->
    /etc/passwd``).  Returns False on any resolution error.
    """
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    try:
        resolved.relative_to(_OUTPUTS_ROOT)
    except ValueError:
        return False
    return True


def render_artifacts_browser() -> None:
    st.divider()
    st.subheader("📦 Artifacts & Outputs")
    if not OUTPUTS_DIR.exists():
        st.info("No artifacts yet. Run an analysis first.")
        return

    run_dirs = sorted(d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_"))
    if not run_dirs:
        st.info("No run directories found.")
        return

    run_path = run_dirs[-1]
    if not _is_inside_outputs(run_path):
        st.error("Refusing to browse run directory outside the outputs root.")
        return

    st.info(f"Showing artifacts from **{run_path.name}**")

    # CWE-22: filter out anything (file or symlink target) that resolves
    # outside the trusted outputs root.
    files = sorted(
        f for f in run_path.rglob("*")
        if f.is_file() and _is_inside_outputs(f)
    )
    if not files:
        st.warning("No files in this directory.")
        return

    for file_path in files:
        rel = file_path.relative_to(run_path)
        with st.expander(f"📄 {rel}", expanded=False):
            col_view, col_dl = st.columns([1, 1])
            with col_view:
                suf = file_path.suffix.lower()
                if suf in (".png", ".jpg", ".jpeg", ".svg"):
                    st.image(str(file_path))
                elif suf == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        st.json(json.load(f))
                elif suf == ".csv":
                    st.dataframe(pd.read_csv(file_path), use_container_width=True)
                elif suf == ".pkl":
                    st.text("Binary model file (download to inspect locally).")
                else:
                    st.text("No preview for this file type.")
            with col_dl:
                with open(file_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        data=f,
                        file_name=file_path.name,
                        mime="application/octet-stream",
                        key=f"dl_{run_path.name}_{rel}",
                    )


# ---------------------------------------------------------------------------
# UI: model comparison chart
# ---------------------------------------------------------------------------

def render_comparison_chart(history: list, task_type: str = "classification") -> None:
    """FIX: added Plotly bar chart for model comparison instead of broken st.markdown."""
    if not history:
        return

    if task_type == "classification":
        metric_keys = {"Accuracy": "accuracy", "F1-Score": "f1"}
    else:
        metric_keys = {"R² Score": "r2", "MSE": "mse"}

    rows = []
    for entry in history:
        for label, key in metric_keys.items():
            rows.append({
                "Model": entry["model"].replace("_Specialist", ""),
                "Metric": label,
                "Value": entry["metrics"].get(key, 0.0),
            })

    fig = px.bar(
        pd.DataFrame(rows),
        x="Model", y="Value", color="Metric",
        barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_range=[0, 1] if task_type == "classification" else None)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Step 1 — Dataset selection
# ---------------------------------------------------------------------------

def step_dataset() -> None:
    render_section_header(
        "Select a dataset",
        "Pick an existing CSV, upload your own, or generate a built-in sample.",
    )

    datasets = get_available_datasets()
    if datasets:
        st.caption(f"{len(datasets)} dataset(s) available")
        cols = st.columns(min(3, len(datasets)))
        for idx, ds in enumerate(datasets):
            with cols[idx % len(cols)]:
                st.button(
                    f"{ds['name']}\n{ds['cols']} columns",
                    key=f"load_{ds['name']}",
                    on_click=set_data_and_advance,
                    args=(ds["path"],),
                    use_container_width=True,
                )
    else:
        st.info("No datasets found in `data/tabular`. Upload a CSV or create a sample below.")

    st.divider()

    tab_upload, tab_create = st.tabs(["Upload CSV", "Generate sample"])

    # --- Upload ---
    with tab_upload:
        st.markdown(
            "Drop a CSV file. The first row is treated as headers; encoding, "
            "separator and column names are normalised automatically."
        )
        with st.expander("CSV requirements & auto-formatting", expanded=False):
            st.markdown(
                "**Requirements**  \n"
                "- First row contains column headers  \n"
                "- At least one target column  \n"
                "- Minimum 10 rows  \n\n"
                "**Auto-formatting applied**  \n"
                "- Encoding detection (UTF-8, Windows-1251, Latin-1, …)  \n"
                "- Separator detection (`,` `;` `\\t` `|`)  \n"
                "- Column-name sanitisation  \n"
                "- Empty row/column removal  \n"
                "- Numeric string conversion (logged)  \n"
                "- Common null-value normalisation"
            )
        uploaded = st.file_uploader("CSV file", type=["csv"], key="upload_csv")
        if uploaded is not None:
            safe_name = _sanitize_filename(uploaded.name)
            save_path = DATA_DIR / safe_name
            if save_path.exists():
                stem = save_path.stem
                save_path = DATA_DIR / f"{stem}_{datetime.now().strftime('%H%M%S')}.csv"

            with st.spinner("Auto-formatting…"):
                success, message = format_and_save_csv(uploaded, save_path)
            if success:
                st.success(message)
                st.cache_data.clear()
                try:
                    st.dataframe(pd.read_csv(save_path, nrows=10), use_container_width=True)
                    set_data_and_advance(save_path)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not read formatted file: {exc}")
            else:
                st.error(message)

    # --- Create test datasets ---
    with tab_create:
        st.markdown("Generate a built-in dataset for quick experimentation.")
        gen = DataGenerator(data_dir=Path("data"))
        creators = {
            "Iris":      ("150 rows, 3 classes",       gen.create_iris_dataset),
            "Wine":      ("178 rows, 3 classes",       gen.create_wine_dataset),
            "Digits":    ("1,797 rows, 10 classes",    gen.create_digits_dataset),
            "Titanic":   ("891 rows, binary target",   gen.create_titanic_dataset),
            "Synthetic": ("500 rows, 3 clusters",      gen.create_synthetic_dataset),
            "Large":     ("100,000 rows, 5 classes",   gen.create_large_dataset),
        }
        cols = st.columns(3)
        for i, (ds_name, (descr, creator)) in enumerate(creators.items()):
            with cols[i % 3]:
                if st.button(
                    f"{ds_name}\n{descr}",
                    key=f"create_{ds_name}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Generating {ds_name}…"):
                        try:
                            path = creator()
                            st.cache_data.clear()
                            set_data_and_advance(path)
                            st.rerun()
                        except Exception as create_exc:
                            st.error(f"Failed to create dataset: {create_exc}")


# ---------------------------------------------------------------------------
# Step 2 — Pipeline configuration
# ---------------------------------------------------------------------------

def step_pipeline() -> None:
    render_section_header(
        "Configure the pipeline",
        "Pick a preset or fine-tune every model and weight.",
    )

    if st.session_state.data_preview is None:
        st.warning("Select a dataset first.")
        if st.button("Back to data selection"):
            st.session_state.step = 0
            st.rerun()
        return

    st.caption(f"Dataset: {st.session_state.data_path.name}")
    with st.expander("Preview (first 100 rows)", expanded=False):
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
        help="Column the model should learn to predict.",
    )
    st.session_state.target_column = target_col

    st.divider()

    if st.session_state.config_state is None:
        load_preset_config("Standard")

    preset_mode = st.radio(
        "Mode",
        ["Fast", "Standard", "Accurate", "Advanced"],
        index=1,
        horizontal=True,
        key="preset_mode_radio",
        captions=[
            "Two models, quick scan",
            "Five models, balanced",
            "Five models with CV",
            "Manual fine-tuning",
        ],
    )

    if preset_mode != "Advanced":
        if (
            st.session_state.get("last_preset") != preset_mode
            or st.session_state.config_state is None
        ):
            load_preset_config(preset_mode)
            st.session_state.last_preset = preset_mode

        cfg = st.session_state.config_state
        enabled = sum(1 for m in cfg["models"].values() if m["enabled"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Models", enabled)
        m2.metric("CV folds", cfg["training"].get("cv_folds", 5))
        m3.metric("Test size", f"{cfg['training'].get('test_size', 0.2):.0%}")
        m4.metric(
            "Scoring",
            f"acc {cfg['scoring']['accuracy_weight']:.2f} / "
            f"f1 {cfg['scoring']['f1_weight']:.2f}",
        )
    else:
        st.caption("Manually tune training, scoring and per-model parameters.")

        if not st.session_state.get("config_initialized"):
            load_preset_config("Accurate")
            st.session_state.config_initialized = True

        with st.form("advanced_config_form"):
            cfg_s = st.session_state.config_state

            st.markdown("##### Training")
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                cv_folds = st.number_input(
                    "CV folds",
                    value=cfg_s["training"].get("cv_folds", 5),
                    min_value=2, max_value=20,
                )
            with tcol2:
                test_size = st.slider(
                    "Test size", 0.1, 0.4,
                    value=cfg_s["training"].get("test_size", 0.2), step=0.05,
                )
            with tcol3:
                use_cv = st.checkbox(
                    "Include CV in final score",
                    value=cfg_s["training"].get("use_cv_in_scoring", False),
                )

            st.markdown("##### Global scoring weights")
            c1, c2, c3 = st.columns(3)
            with c1:
                aw = st.slider("Accuracy", 0.0, 1.0,
                               value=cfg_s["scoring"].get("accuracy_weight", 0.7), step=0.05)
            with c2:
                fw = st.slider("F1", 0.0, 1.0,
                               value=cfg_s["scoring"].get("f1_weight", 0.3), step=0.05)
            with c3:
                cw = st.slider("CV", 0.0, 1.0,
                               value=cfg_s["scoring"].get("cv_weight", 0.0), step=0.05)

            st.markdown("##### Per-model configuration")

            model_updates: dict = {}
            for model_key, model_data in cfg_s["models"].items():
                with st.expander(model_data["name"], expanded=False):
                    col_en, col_desc = st.columns([1, 3])
                    with col_en:
                        enabled = st.checkbox(
                            "Enabled",
                            value=model_data.get("enabled", True),
                            key=f"en_{model_key}",
                        )
                    with col_desc:
                        st.caption(model_data.get("description", ""))

                    st.markdown("**Hyperparameters (JSON)**")
                    params_json = json.dumps(model_data.get("params", {}), indent=2)
                    raw_params_str = st.text_area(
                        "Params", value=params_json, height=120, key=f"p_{model_key}",
                        label_visibility="collapsed",
                    )

                    try:
                        parsed_params = json.loads(raw_params_str)
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid JSON: {exc}")
                        parsed_params = model_data.get("params", {})

                    st.markdown("**Custom scoring weights** (override global)")
                    wc1, wc2, wc3 = st.columns(3)
                    existing_w = model_data.get("custom_scoring_weights", {})
                    with wc1:
                        m_aw = st.slider(
                            "Accuracy", 0.0, 1.0,
                            value=existing_w.get(
                                "accuracy_weight", cfg_s["scoring"]["accuracy_weight"]
                            ),
                            step=0.05, key=f"aw_{model_key}",
                        )
                    with wc2:
                        m_fw = st.slider(
                            "F1", 0.0, 1.0,
                            value=existing_w.get(
                                "f1_weight", cfg_s["scoring"]["f1_weight"]
                            ),
                            step=0.05, key=f"fw_{model_key}",
                        )
                    with wc3:
                        m_cw = st.slider(
                            "CV", 0.0, 1.0,
                            value=existing_w.get(
                                "cv_weight", cfg_s["scoring"]["cv_weight"]
                            ),
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

            submitted = st.form_submit_button(
                "Apply configuration", type="primary", use_container_width=True
            )
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

    st.divider()
    col_back, col_run = st.columns([1, 2])
    with col_back:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    with col_run:
        if st.button("Start training", type="primary", use_container_width=True):
            try:
                config_obj = build_config_from_state()
                st.session_state.pipeline_config = config_obj
                st.session_state.step = 2
                st.rerun()
            except ValueError as exc:
                st.error(f"Configuration error: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — Training
# ---------------------------------------------------------------------------

def step_training() -> None:
    render_section_header(
        "Train models",
        "Each enabled model is fitted on the train split and scored on the holdout.",
    )

    if st.session_state.data_path is None or st.session_state.pipeline_config is None:
        st.warning("Missing data or configuration.")
        if st.button("Back to configuration"):
            st.session_state.step = 1
            st.rerun()
        return

    if st.session_state.training_complete:
        st.success("Training complete.")
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
        # 1. Load data
        status_text.text("Loading dataset…")
        progress_bar.progress(10)
        data = pd.read_csv(st.session_state.data_path)
        config = st.session_state.pipeline_config

        # 2. Resolve target column (set in step_pipeline)
        target_column = st.session_state.get("target_column")
        if target_column is None or target_column not in data.columns:
            # Fallback auto-detection
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
        y = data[target_column].values
        feature_columns = [
            c for c in data.columns
            if c != target_column and pd.api.types.is_numeric_dtype(data[c])
        ]
        st.session_state.feature_columns = feature_columns
        X = data[feature_columns].values

        # 3. Profile
        status_text.text("Profiling data…")
        progress_bar.progress(25)
        profiler = DataProfiler(dataset_name=st.session_state.data_path.stem)
        profile = profiler.profile_tabular_data(X, y, feature_names=feature_columns)
        st.session_state.profile = profile
        task_type = profile.task_type

        # 4. Register models
        status_text.text("Preparing models…")
        progress_bar.progress(40)
        selector = AdaptiveModelSelector()
        selector.register_models_from_pipeline_config(config, task_type=task_type)

        # 5. Train & select — real per-model progress via callback.
        # FIX: previously this loop only animated the bar without doing any
        # work, then training ran in one blocking call. Now the bar advances
        # as each model finishes.
        def _on_model(idx: int, total: int, name: str) -> None:
            pct = 40 + int(50 * (idx - 1) / max(total, 1))
            status_text.text(f"Training {name} ({idx}/{total})…")
            progress_bar.progress(min(pct, 90))

        best_model, X_test, y_test = selector.profile_and_select(
            X, y,
            data_profile=profile.to_dict(),
            cv_folds=config.training.get("cv_folds", 5),
            test_size=config.training.get("test_size", 0.2),    # FIX: pass from config
            random_state=config.training.get("random_state", 42),
            use_cv_in_scoring=config.training.get("use_cv_in_scoring", False),
            global_accuracy_weight=config.scoring.get("accuracy_weight", 0.7),
            global_f1_weight=config.scoring.get("f1_weight", 0.3),
            global_cv_weight=config.scoring.get("cv_weight", 0.0),
            progress_cb=_on_model,
        )

        st.session_state.best_model = best_model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        if hasattr(selector, "selection_history") and selector.selection_history:
            st.session_state.model_comparison_history = selector.selection_history

        # 6. Build results dict (task-type aware)
        metrics = best_model.performance_metrics
        if task_type == "classification":
            primary_metric = {"accuracy": float(metrics.get("accuracy", 0)), "f1": float(metrics.get("f1", 0))}
        else:
            primary_metric = {"r2": float(metrics.get("r2", 0)), "rmse": float(metrics.get("rmse", 0))}

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
        status_text.text("Saving artefacts…")
        save_results_to_session(output_dir, session_id, st.session_state.results,
                                profile, best_model, selector, X_test, y_test)
        progress_bar.progress(100)
        status_text.empty()

        st.session_state.training_complete = True
        st.session_state.step = 3
        st.rerun()

    except Exception as exc:
        # FIX: log full traceback; show user-friendly message only
        logger.exception("Training failed")
        st.session_state.error_msg = f"Training error: {type(exc).__name__}: {exc}"
        st.session_state.error_detail = traceback.format_exc()
        st.rerun()


# ---------------------------------------------------------------------------
# Step 4 — Results
# ---------------------------------------------------------------------------

def step_results() -> None:
    render_section_header(
        "Results",
        f"Session {st.session_state.session_id:03d}"
        if st.session_state.session_id else "",
    )

    if not st.session_state.training_complete or st.session_state.results is None:
        st.warning("No results available.")
        if st.button("Back to training"):
            st.session_state.step = 2
            st.rerun()
        return

    res = st.session_state.results
    task_type: str = res.get("task_type", "classification")

    # --- Key metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best model", res["best_model"].replace("_Specialist", ""))
    if task_type == "classification":
        with col2:
            st.metric("Accuracy", f"{res.get('accuracy', 0):.4f}")
        with col3:
            st.metric("F1-score", f"{res.get('f1', 0):.4f}")
    else:
        with col2:
            st.metric("R squared", f"{res.get('r2', 0):.4f}")
        with col3:
            st.metric("RMSE", f"{res.get('rmse', 0):.4f}")

    # --- Model comparison ---
    if st.session_state.model_comparison_history:
        history = st.session_state.model_comparison_history
        st.markdown("##### Model comparison")
        render_comparison_chart(history, task_type)

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

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Data profile summary ---
    if st.session_state.profile:
        with st.expander("Data profile", expanded=False):
            p = st.session_state.profile.to_dict()
            st.json({
                "n_samples": p["n_samples"],
                "n_features": p["n_features"],
                "n_classes": p.get("n_classes"),
                "task_type": p.get("task_type"),
                "data_complexity": p["data_complexity"],
                "recommended_models": p["recommended_models"],
                "preprocessing_needs": p["preprocessing_needs"],
            })

    st.divider()

    # --- Download buttons ---
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        report_json = json.dumps(res, indent=2, ensure_ascii=False)
        st.download_button(
            "Download report (JSON)",
            data=report_json,
            file_name="astra_report.json",
            mime="application/json",
            use_container_width=True,
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
                use_container_width=True,
            )

    render_artifacts_browser()

    if st.session_state.output_dir:
        st.caption(f"Artifacts saved to: {st.session_state.output_dir}")

    if st.button("Start new analysis", type="primary", use_container_width=True):
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="astra-brand" style="padding:8px 0 16px 0;">
                <div class="astra-logo">A</div>
                <div>
                    <div class="astra-title" style="font-size:18px;">Astra ML</div>
                    <div class="astra-sub" style="font-size:11px;">v2.7.5</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current = st.session_state.step
        st.markdown("##### Steps")
        for i, label in enumerate(_STEP_LABELS):
            disabled = i > current  # cannot jump ahead of progress
            prefix = "•" if i == current else " "
            if st.button(
                f"{prefix}  {i + 1}. {label}",
                key=f"nav_{i}",
                disabled=disabled,
                use_container_width=True,
            ):
                st.session_state.step = i
                st.rerun()

        st.divider()
        if st.button("Reset session", use_container_width=True):
            reset_session()
            st.rerun()

        st.caption("Local-only. Data never leaves this machine.")


def main() -> None:
    init_state()
    init_directories()
    inject_styles()

    _render_sidebar()
    render_brand()

    # FIX (CWE-209): show a user-friendly message only. Full traceback is
    # written to logs unconditionally, and shown in the UI ONLY when
    # ASTRA_DEBUG=1 is set in the environment.
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        if DEBUG_MODE and st.session_state.get("error_detail"):
            with st.expander("Technical details (debug mode)", expanded=False):
                st.code(st.session_state.error_detail)
        if st.button("Clear error and restart"):
            reset_session()
            st.rerun()
        return

    step = st.session_state.step
    render_stepper(step)

    if step == 0:
        step_dataset()
    elif step == 1:
        step_pipeline()
    elif step == 2:
        step_training()
    elif step == 3:
        step_results()


if __name__ == "__main__":
    main()
