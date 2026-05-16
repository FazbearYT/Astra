import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import sys
import traceback
import matplotlib
import chardet
import re
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
script_path = Path(__file__).parent
sys.path.insert(0, str(script_path))
from src.model_profiler import DataProfiler
from src.model_selector import AdaptiveModelSelector
from src.pipeline_config import get_default_config, get_fast_config, get_accurate_config, PipelineConfig, ModelConfig
from sklearn.metrics import accuracy_score, f1_score
st.set_page_config(page_title="Astra ML System", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")
def init_state():
    defaults = {
        'step': 0,
        'data_path': None,
        'data_preview': None,
        'results': None,
        'profile': None,
        'best_model': None,
        'X_test': None,
        'y_test': None,
        'pipeline_config': None,
        'training_complete': False,
        'error_msg': None,
        'target_column': None,
        'feature_columns': [],
        'config_state': None,
        'model_comparison_history': None,
        'advanced_mode': False,
        'session_id': None,
        'output_dir': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
def reset_session():
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    init_state()
def init_directories():
    Path("data/tabular").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
def setup_output_directory():
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    existing_runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if existing_runs:
        last_num = int(existing_runs[-1].name.split("_")[1])
        session_id = last_num + 1
    else:
        session_id = 1
    output_dir = outputs_dir / f"run_{session_id:03d}"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    return session_id, output_dir
def get_available_datasets():
    tabular_dir = Path("data/tabular")
    datasets = []
    if tabular_dir.exists():
        for csv_file in tabular_dir.glob("*.csv"):
            try:
                df_peek = pd.read_csv(csv_file, nrows=1)
                datasets.append({'name': csv_file.stem, 'path': csv_file, 'cols': len(df_peek.columns)})
            except Exception:
                pass
    return datasets
def create_test_dataset(name: str) -> Path:
    from sklearn.datasets import load_iris, load_wine, load_digits, make_blobs, make_classification
    tabular_dir = Path("data/tabular")
    df = pd.DataFrame()
    filepath = None
    if name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        filepath = tabular_dir / "iris.csv"
    elif name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        filepath = tabular_dir / "wine.csv"
    elif name == "Digits":
        data = load_digits()
        df = pd.DataFrame(data.data, columns=[f'pixel_{i}' for i in range(64)])
        df['target'] = data.target
        filepath = tabular_dir / "digits.csv"
    elif name == "Titanic":
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            'pclass': np.random.randint(1, 4, n),
            'gender': np.random.randint(0, 2, n),
            'age': np.random.normal(30, 14, n).clip(0, 80),
            'sibsp': np.random.poisson(0.5, n),
            'parch': np.random.poisson(0.4, n),
            'fare': np.random.exponential(30, n),
        })
        df['target'] = ((df['gender'] == 1) & (df['pclass'] < 3) | (df['age'] < 10)).astype(int)
        filepath = tabular_dir / "titanic.csv"
    elif name == "Synthetic":
        X, y = make_blobs(n_samples=500, n_features=5, centers=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        filepath = tabular_dir / "synthetic.csv"
    elif name == "Large":
        with st.spinner("Generating large dataset (100K rows)..."):
            X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            df['target'] = y
            filepath = tabular_dir / "large_test.csv"
    if filepath and not df.empty:
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath
    return None
def detect_encoding(file_path_or_buffer) -> str:
    if isinstance(file_path_or_buffer, (str, Path)):
        with open(file_path_or_buffer, 'rb') as f:
            raw_data = f.read(10000)
    else:
        raw_data = file_path_or_buffer.read(10000)
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
    result = chardet.detect(raw_data)
    return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
def detect_separator(df_sample: pd.DataFrame) -> str:
    if len(df_sample.columns) > 1:
        return ','
    return ','
def auto_format_dataframe(df: pd.DataFrame, filename: str = "uploaded") -> pd.DataFrame:
    df_clean = df.copy()
    original_cols = df_clean.columns.tolist()
    cleaned_cols = []
    seen_cols = {}
    for col in original_cols:
        col_str = str(col).strip()
        col_clean = re.sub(r'[^\w\u0400-\u04FF]', '_', col_str, flags=re.UNICODE)
        col_clean = col_clean.strip('_')
        if not col_clean:
            col_clean = f"unnamed_{len(cleaned_cols)}"
        if col_clean in seen_cols:
            seen_cols[col_clean] += 1
            col_clean = f"{col_clean}_{seen_cols[col_clean]}"
        else:
            seen_cols[col_clean] = 0
        cleaned_cols.append(col_clean)
    df_clean.columns = cleaned_cols
    df_clean = df_clean.dropna(how='all', axis=0)
    df_clean = df_clean.dropna(how='all', axis=1)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
    df_clean = df_clean.replace(['', ' ', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN'], np.nan)
    if len(df_clean.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns after cleaning")
    df_clean = df_clean.dropna(how='all')
    return df_clean
def format_and_save_csv(uploaded_file, save_path: Path, filename: str = None) -> tuple[bool, str]:
    try:
        if filename is None:
            filename = getattr(uploaded_file, 'name', 'uploaded_data.csv')
        encoding = detect_encoding(uploaded_file)
        uploaded_file.seek(0)
        df = None
        separator_used = ','
        for sep in [',', ';', '\t', '|']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, nrows=100)
                if len(df.columns) > 1:
                    separator_used = sep
                    break
            except:
                continue
        if df is None or len(df.columns) <= 1:
            return False, "Could not parse CSV file. Please check the file format."
        uploaded_file.seek(0)
        df_full = pd.read_csv(uploaded_file, encoding=encoding, sep=separator_used)
        df_formatted = auto_format_dataframe(df_full, filename)
        if len(df_formatted) < 10:
            return False, f"Dataset has only {len(df_formatted)} rows. Minimum 10 rows recommended."
        if len(df_formatted.columns) < 2:
            return False, "Dataset must have at least 2 columns (features + target)."
        df_formatted.to_csv(save_path, index=False, encoding='utf-8')
        return True, f"Successfully formatted and saved. Used separator: '{separator_used}', Encoding: {encoding}"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"
def set_data_and_advance(path: Path):
    try:
        st.session_state.data_path = path
        st.session_state.data_preview = pd.read_csv(path, nrows=100)
        st.session_state.step = 1
    except Exception as e:
        st.session_state.error_msg = f"Failed to read {path.name}: {e}"
def load_preset_config(preset: str):
    if preset == "Fast":
        cfg = get_fast_config()
    elif preset == "Accurate":
        cfg = get_accurate_config()
    else:
        cfg = get_default_config()
    st.session_state.config_state = {
        'training': cfg.training.copy(),
        'scoring': cfg.scoring.copy(),
        'models': {k: {
            'name': v.name,
            'enabled': v.enabled,
            'params': v.params,
            'profile_requirements': v.profile_requirements,
            'description': v.description,
            'custom_scoring_weights': v.custom_scoring_weights if v.custom_scoring_weights else {
                "accuracy_weight": cfg.scoring["accuracy_weight"], "f1_weight": cfg.scoring["f1_weight"],
                "cv_weight": cfg.scoring["cv_weight"]}
        } for k, v in cfg.models.items()}
    }
def build_config_from_state() -> PipelineConfig:
    cfg_data = st.session_state.config_state
    training = cfg_data['training']
    scoring = cfg_data['scoring']
    models = {}
    for k, v in cfg_data['models'].items():
        try:
            params = v['params'] if isinstance(v['params'], dict) else json.loads(v['params'])
        except:
            params = v['params']
        try:
            weights = v['custom_scoring_weights'] if isinstance(v['custom_scoring_weights'], dict) else json.loads(v['custom_scoring_weights'])
        except:
            weights = v['custom_scoring_weights']
        models[k] = ModelConfig(
            name=v['name'],
            enabled=v['enabled'],
            params=params,
            profile_requirements=v['profile_requirements'],
            description=v['description'],
            custom_scoring_weights=weights
        )
    return PipelineConfig(models=models, training=training, scoring=scoring)
def save_visualizations(output_dir: Path, profile, best_model, X_test, y_test):
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    try:
        if profile:
            profile_path = viz_dir / "data_profile.png"
            profile.visualize_profile(save_path=str(profile_path))
            plt.close('all')
    except Exception as e:
        print(f"Error saving data profile visualization: {e}")
    try:
        if best_model and best_model.is_trained:
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            ax.set_title(f'Confusion Matrix - {best_model.name}')
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            cm_path = viz_dir / "confusion_matrix.png"
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            plt.close('all')
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
def save_results_to_session(output_dir: Path, session_id: int, results: dict, profile, best_model, selector, X_test, y_test):
    results_path = output_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if profile:
        profile_path = output_dir / "profile.json"
        try:
            profile.save(str(profile_path))
        except Exception:
            pass
    if selector:
        models_dir = output_dir / "models"
        try:
            selector.save_all_models(str(models_dir))
        except Exception:
            pass
    save_visualizations(output_dir, profile, best_model, X_test, y_test)
def render_artifacts_browser():
    st.divider()
    st.subheader("📦 Artifacts & Outputs")
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        st.info("No artifacts found. Run an analysis to generate outputs.")
        return
    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        st.info("No artifacts found. Run an analysis to generate outputs.")
        return
    run_path = run_dirs[-1]
    st.info(f"Showing artifacts from latest session: **{run_path.name}**")
    files = [f for f in run_path.rglob("*") if f.is_file()]
    if not files:
        st.warning("No files in this directory.")
        return
    for file_path in sorted(files):
        rel_name = file_path.relative_to(run_path)
        with st.expander(f"📄 {rel_name}", expanded=False):
            col_view, col_dl = st.columns([1, 1])
            with col_view:
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                    st.image(str(file_path))
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        st.json(json.load(f))
                elif file_path.suffix.lower() == '.csv':
                    st.dataframe(pd.read_csv(file_path))
                elif file_path.suffix == '.pkl':
                    st.text("Binary model file (download to inspect)")
                else:
                    st.text("Preview not available for this file type")
            with col_dl:
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=file_path.name,
                        mime="application/octet-stream",
                        key=f"dl_{run_path.name}_{rel_name}"
                    )
def main():
    init_state()
    init_directories()
    st.title("Astra ML System")
    st.markdown("Automatic ML model selection based on data profiling")
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        if st.button("Clear error and restart"):
            reset_session()
            st.rerun()
        return
    if st.session_state.step == 0:
        step_dataset()
    elif st.session_state.step == 1:
        step_pipeline()
    elif st.session_state.step == 2:
        step_training()
    elif st.session_state.step == 3:
        step_results()
def step_dataset():
    st.header("Step 1: Select Data")
    st.progress(0.25)
    st.subheader("Available Datasets")
    datasets = get_available_datasets()
    if datasets:
        cols = st.columns(min(3, len(datasets)))
        for idx, ds in enumerate(datasets):
            with cols[idx % len(cols)]:
                st.button(
                    f"📊 {ds['name']}\n({ds['cols']} columns)",
                    key=f"load_{ds['name']}",
                    on_click=set_data_and_advance,
                    args=(ds['path'],),
                    use_container_width=True
                )
    else:
        st.info("No existing datasets found in `data/tabular`. Please upload a CSV file or create a test dataset below.")
    st.divider()
    col_upload, col_create = st.columns(2)
    with col_upload:
        with st.expander("📤 Upload CSV File (Auto-Format Enabled)", expanded=False):
            st.markdown("""
            **CSV Requirements:**
            - First row must contain column headers
            - Must contain at least one target column (binary or multi-class labels)
            - Minimum 10 rows recommended for meaningful analysis

            **Auto-Formatting Features:**
            - ✅ Automatic encoding detection (UTF-8, Windows-1251, Latin-1, etc.)
            - ✅ Automatic separator detection (comma, semicolon, tab, pipe)
            - ✅ Column name cleaning (special characters → underscores)
            - ✅ Duplicate column name handling
            - ✅ Empty row/column removal
            - ✅ Numeric string conversion
            - ✅ Common missing value normalization (NA, N/A, null, etc.)

            **Supported Formats:**
            - Standard CSV files with various encodings
            - Files with different separators (`,`, `;`, `\t`, `|`)
            - Files with non-standard column names
            """)
            uploaded_file = st.file_uploader("Select CSV file", type=["csv"], key="upload_csv")
            if uploaded_file is not None:
                save_path = Path("data/tabular") / uploaded_file.name
                with st.spinner("🔄 Auto-formatting dataset..."):
                    success, message = format_and_save_csv(uploaded_file, save_path)
                    if success:
                        st.success(f"✅ {message}")
                        try:
                            df_preview = pd.read_csv(save_path, nrows=10)
                            st.write("**Preview of formatted data:**")
                            st.dataframe(df_preview)
                            set_data_and_advance(save_path)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error reading formatted file: {e}")
                    else:
                        st.error(f"❌ {message}")
                        st.info("Please ensure your CSV file meets the minimum requirements listed above.")
    with col_create:
        with st.expander("🧪 Create Test Dataset", expanded=False):
            st.markdown("Generate synthetic datasets for testing and demonstration purposes.")
            test_datasets = ["Iris", "Wine", "Digits", "Titanic", "Synthetic", "Large"]
            for ds_name in test_datasets:
                if st.button(f"Create '{ds_name}'", key=f"create_{ds_name}", use_container_width=True):
                    with st.spinner(f"Creating '{ds_name}'..."):
                        path = create_test_dataset(ds_name)
                        if path:
                            set_data_and_advance(path)
                            st.rerun()
def step_pipeline():
    st.header("Step 2: Pipeline Configuration")
    st.progress(0.5)
    if st.session_state.data_preview is None:
        st.warning("Please select data in Step 1 first.")
        if st.button("Back to data selection"):
            st.session_state.step = 0
            st.rerun()
        return
    st.success(f"Dataset: `{st.session_state.data_path.name}`")
    st.dataframe(st.session_state.data_preview)
    if st.session_state.config_state is None:
        load_preset_config("Standard")
    preset_mode = st.radio(
        "Configuration Mode: ",
        ["Fast", "Standard", "Accurate", "Advanced Customization"],
        index=1,
        key="preset_mode_radio"
    )
    if preset_mode != "Advanced Customization":
        if st.session_state.get('last_preset') != preset_mode or st.session_state.config_state is None:
            load_preset_config(preset_mode)
            st.session_state.last_preset = preset_mode
        st.info(f"Using '{preset_mode}' preset configuration")
        cfg = st.session_state.config_state
        st.json({
            "models_enabled": sum(1 for m in cfg['models'].values() if m['enabled']),
            "cv_folds": cfg['training'].get('cv_folds', 5),
            "scoring_weights": cfg['scoring']
        })
    else:
        st.subheader("Advanced Pipeline Configuration")
        st.warning("Advanced mode allows fine-tuning of all parameters")
        if 'config_initialized' not in st.session_state or not st.session_state.config_initialized:
            load_preset_config("Accurate")
            st.session_state.config_initialized = True
        with st.form("advanced_config_form"):
            st.session_state.config_state['training']['cv_folds'] = st.number_input("CV Folds", value=st.session_state.config_state['training'].get('cv_folds', 5), min_value=2, max_value=20)
            st.session_state.config_state['training']['test_size'] = st.slider("Test Size", 0.1, 0.4, value=st.session_state.config_state['training'].get('test_size', 0.2), step=0.05)
            st.session_state.config_state['training']['use_cv_in_scoring'] = st.checkbox("Use CV in Final Scoring", value=st.session_state.config_state['training'].get('use_cv_in_scoring', False))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.config_state['scoring']['accuracy_weight'] = st.slider("Global Accuracy Weight", 0.0, 1.0, value=st.session_state.config_state['scoring'].get('accuracy_weight', 0.7), step=0.05)
            with col2:
                st.session_state.config_state['scoring']['f1_weight'] = st.slider("Global F1 Weight", 0.0, 1.0, value=st.session_state.config_state['scoring'].get('f1_weight', 0.3), step=0.05)
            with col3:
                st.session_state.config_state['scoring']['cv_weight'] = st.slider("Global CV Weight", 0.0, 1.0, value=st.session_state.config_state['scoring'].get('cv_weight', 0.0), step=0.05)
            st.divider()
            st.markdown("### Model-Specific Configuration")
            for model_key, model_data in st.session_state.config_state['models'].items():
                with st.expander(f"🔧 {model_data['name']}", expanded=False):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        model_data['enabled'] = st.checkbox("Enabled", value=model_data.get('enabled', True), key=f"en_{model_key}")
                    with c2:
                        st.markdown(f"*{model_data.get('description', '')}*")
                    st.markdown("**Hyperparameters (JSON)**")
                    params_json = json.dumps(model_data.get('params', {}), indent=2)
                    try:
                        model_data['params'] = json.loads(st.text_area("Params", value=params_json, height=100, key=f"p_{model_key}"))
                    except:
                        st.error("Invalid JSON for params")
                    st.markdown("**Custom Scoring Weights** (Overrides global if provided)")
                    wc1, wc2, wc3 = st.columns(3)
                    with wc1:
                        if 'custom_scoring_weights' not in model_data:
                            model_data['custom_scoring_weights'] = {}
                        model_data['custom_scoring_weights']['accuracy_weight'] = st.slider("Acc", 0.0, 1.0, value=model_data.get('custom_scoring_weights', {}).get('accuracy_weight', st.session_state.config_state['scoring']['accuracy_weight']), step=0.05, key=f"aw_{model_key}")
                    with wc2:
                        model_data['custom_scoring_weights']['f1_weight'] = st.slider("F1", 0.0, 1.0, value=model_data.get('custom_scoring_weights', {}).get('f1_weight', st.session_state.config_state['scoring']['f1_weight']), step=0.05, key=f"fw_{model_key}")
                    with wc3:
                        model_data['custom_scoring_weights']['cv_weight'] = st.slider("CV", 0.0, 1.0, value=model_data.get('custom_scoring_weights', {}).get('cv_weight', st.session_state.config_state['scoring']['cv_weight']), step=0.05, key=f"cw_{model_key}")
            form_submitted = st.form_submit_button("Apply Advanced Configuration", type="primary")
            if form_submitted:
                st.success("Advanced configuration applied!")
    col_back, col_run = st.columns(2)
    with col_back:
        if st.button("Back"):
            st.session_state.step = 0
            st.rerun()
    with col_run:
        if st.button("Save & Start Training", type="primary"):
            try:
                config_obj = build_config_from_state()
                st.session_state.pipeline_config = config_obj
                st.session_state.step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Configuration Error: {e}")
def step_training():
    st.header("Step 3: Model Training")
    st.progress(0.75)
    if st.session_state.data_path is None or st.session_state.pipeline_config is None:
        st.warning("Missing data or configuration. Go back.")
        if st.button("Back to configuration"):
            st.session_state.step = 1
            st.rerun()
        return
    if st.session_state.training_complete:
        st.success("Training completed successfully.")
        if st.button("View Results", type="primary"):
            st.session_state.step = 3
            st.rerun()
        return
    try:
        session_id, output_dir = setup_output_directory()
        st.session_state.session_id = session_id
        st.session_state.output_dir = output_dir
        with st.spinner("Analyzing and training... This may take a while."):
            status_text = st.empty()
            status_text.text("Loading full dataset...")
            data = pd.read_csv(st.session_state.data_path)
            config = st.session_state.pipeline_config
            target_candidates = ['target', 'class', 'label', 'category', 'target_name']
            target_column = next((col for col in target_candidates if col in data.columns), None)
            if target_column is None:
                numeric_cols = data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 1:
                    target_column = numeric_cols[-1]
                else:
                    raise ValueError("No target column found.")
            st.session_state.target_column = target_column
            y = data[target_column].values
            feature_columns = [col for col in data.columns if col != target_column and pd.api.types.is_numeric_dtype(data[col])]
            st.session_state.feature_columns = feature_columns
            X = data[feature_columns].values
            status_text.text("Profiling data...")
            profiler = DataProfiler(dataset_name=st.session_state.data_path.stem)
            profile = profiler.profile_tabular_data(X, y, feature_names=feature_columns)
            st.session_state.profile = profile
            selector = AdaptiveModelSelector()
            selector.register_models_from_pipeline_config(config)
            status_text.text(f"Training {len(selector.models)} models...")
            best_model, X_test, y_test = selector.profile_and_select(
                X, y, data_profile=profile.to_dict(),
                cv_folds=config.training.get('cv_folds', 5),
                use_cv_in_scoring=config.training.get('use_cv_in_scoring', False),
                global_accuracy_weight=config.scoring.get('accuracy_weight', 0.7),
                global_f1_weight=config.scoring.get('f1_weight', 0.3),
                global_cv_weight=config.scoring.get('cv_weight', 0.0)
            )
            st.session_state.best_model = best_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            if hasattr(selector, 'selection_history') and selector.selection_history:
                st.session_state.model_comparison_history = selector.selection_history
            acc = best_model.performance_metrics.get('accuracy', 0.0)
            f1 = best_model.performance_metrics.get('f1', 0.0)
            st.session_state.results = {
                'session_id': session_id,
                'accuracy': float(acc), 'f1': float(f1), 'best_model': best_model.name,
                'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1]),
                'timestamp': str(datetime.now())
            }
            save_results_to_session(output_dir, session_id, st.session_state.results, profile, best_model, selector, X_test, y_test)
            st.session_state.training_complete = True
            st.session_state.step = 3
            st.rerun()
    except Exception as e:
        st.session_state.error_msg = f"Training Error: {e}\n{traceback.format_exc()}"
        st.rerun()
def step_results():
    st.header("Step 4: Results")
    st.progress(1.0)
    if not st.session_state.training_complete or st.session_state.results is None:
        st.warning("No results available.")
        if st.button("Back to training"):
            st.session_state.step = 2
            st.rerun()
        return
    res = st.session_state.results
    if st.session_state.session_id:
        st.info(f"✅ Session **{st.session_state.session_id:03d}** completed successfully!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", res['best_model'])
    with col2:
        st.metric("🎯 Accuracy", f"{res['accuracy']:.4f}")
    with col3:
        st.metric("📊 F1-Score", f"{res.get('f1', 0.0):.4f}")
    if st.session_state.profile:
        st.subheader("Data Profile")
        p = st.session_state.profile.to_dict()
        st.json({
            "n_samples": p['n_samples'],
            "n_features": p['n_features'],
            "n_classes": p['n_classes'],
            "data_complexity": p['data_complexity'],
            "recommended_models": p['recommended_models'],
            "preprocessing_needs": p['preprocessing_needs']
        })
    if st.session_state.model_comparison_history:
        with st.expander("📊 Detailed Model Comparison"):
            history = st.session_state.model_comparison_history
            comparison_data = []
            for entry in history:
                comparison_data.append({
                    'Model': entry['model'],
                    'Accuracy': f"{entry['metrics']['accuracy']:.4f}",
                    'F1-Score': f"{entry['metrics']['f1']:.4f}",
                    'Precision': f"{entry['metrics']['precision']:.4f}",
                    'Recall': f"{entry['metrics']['recall']:.4f}",
                    'Score': f"{entry['final_score']:.4f}",
                    'Profile Score': f"{entry['profile_score']:.2f}",
                    'CV Mean': f"{entry['cv_results']['cv_mean']:.4f}" if 'cv_results' in entry else "-"
                })
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True)
    render_artifacts_browser()
    if st.session_state.output_dir:
        st.success(f"📁 All artifacts saved to: **{st.session_state.output_dir}**")
    if st.button("Start New Analysis", type="primary"):
        reset_session()
        st.rerun()
if __name__ == "__main__":
    main()