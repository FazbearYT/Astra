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
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')
script_path = Path(__file__).parent
sys.path.insert(0, str(script_path))

from src.data_generator import DataGenerator
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
        'advanced_mode': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    logger.debug("Session state initialized")

def reset_session():
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    init_state()
    logger.info("Session reset")

def init_directories():
    Path("data/tabular").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    logger.debug("Directories initialized")

def get_available_datasets():
    generator = DataGenerator()
    datasets = generator.auto_detect_datasets()
    logger.info(f"Found {len(datasets)} available datasets")
    return datasets

def create_test_dataset(name: str) -> Path:
    logger.info(f"Creating test dataset: {name}")
    generator = DataGenerator()
    if name == "Iris":
        path = generator.create_iris_dataset()
    elif name == "Wine":
        path = generator.create_wine_dataset()
    elif name == "Digits":
        path = generator.create_digits_dataset()
    elif name == "Titanic":
        path = generator.create_titanic_dataset()
    elif name == "Synthetic":
        path = generator.create_synthetic_dataset()
    elif name == "Large":
        with st.spinner("Generating large dataset (100K rows)..."):
            path = generator.create_large_dataset()
    else:
        logger.error(f"Unknown dataset name: {name}")
        return None
    logger.info(f"Test dataset created: {path}")
    return path

def detect_encoding(file_path_or_buffer) -> str:
    if isinstance(file_path_or_buffer, (str, Path)):
        with open(file_path_or_buffer, 'rb') as f:
            raw_data = f.read(10000)
    else:
        raw_data = file_path_or_buffer.read(10000)
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
    result = chardet.detect(raw_data)
    encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    logger.debug(f"Detected encoding: {encoding}")
    return encoding

def detect_separator(df_sample: pd.DataFrame) -> str:
    if len(df_sample.columns) > 1:
        return ','
    return ','

def auto_format_dataframe(df: pd.DataFrame, filename: str = "uploaded") -> pd.DataFrame:
    logger.info(f"Auto-formatting dataframe: {filename}")
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
    logger.info(f"Dataframe formatted: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    return df_clean

def format_and_save_csv(uploaded_file, save_path: Path, filename: str = None) -> tuple:
    logger.info(f"Formatting and saving CSV: {filename}")
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
            logger.error("Could not parse CSV file")
            return False, "Could not parse CSV file. Please check the file format."

        uploaded_file.seek(0)
        df_full = pd.read_csv(uploaded_file, encoding=encoding, sep=separator_used)
        df_formatted = auto_format_dataframe(df_full, filename)

        if len(df_formatted) < 10:
            logger.error(f"Dataset too small: {len(df_formatted)} rows")
            return False, f"Dataset has only {len(df_formatted)} rows. Minimum 10 rows recommended."

        if len(df_formatted.columns) < 2:
            logger.error("Dataset must have at least 2 columns")
            return False, "Dataset must have at least 2 columns (features + target)."

        df_formatted.to_csv(save_path, index=False, encoding='utf-8')
        logger.info(f"CSV formatted and saved: {save_path}")
        return True, f"Successfully formatted and saved. Used separator: '{separator_used}', Encoding: {encoding}"

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return False, f"Error processing file: {str(e)}"

def set_data_and_advance(path: Path):
    try:
        st.session_state.data_path = path
        st.session_state.data_preview = pd.read_csv(path, nrows=100)
        st.session_state.step = 1
        logger.info(f"Data loaded and advanced to step 1: {path.name}")
    except Exception as e:
        error_msg = f"Failed to read {path.name}: {e}"
        st.session_state.error_msg = error_msg
        logger.error(error_msg)

def load_preset_config(preset: str):
    logger.info(f"Loading preset configuration: {preset}")
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
                "accuracy_weight": cfg.scoring["accuracy_weight"],
                "f1_weight": cfg.scoring["f1_weight"],
                "cv_weight": cfg.scoring["cv_weight"]
            }
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
    logger.info(f"Built pipeline config with {len(models)} models")
    return PipelineConfig(models=models, training=training, scoring=scoring)

def main():
    init_state()
    init_directories()
    logger.info("Astra ML System started")
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
                            logger.error(f"Error reading formatted file: {e}")
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
            st.session_state.config_state['training']['cv_folds'] = st.number_input("CV Folds",
                                                                                    value=st.session_state.config_state['training'].get('cv_folds', 5),
                                                                                    min_value=2, max_value=20)
            st.session_state.config_state['training']['test_size'] = st.slider("Test Size", 0.1, 0.4,
                                                                               value=st.session_state.config_state['training'].get('test_size', 0.2),
                                                                               step=0.05)
            st.session_state.config_state['training']['use_cv_in_scoring'] = st.checkbox("Use CV in Final Scoring",
                                                                                         value=st.session_state.config_state['training'].get('use_cv_in_scoring', False))

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
                logger.error(f"Configuration error: {e}", exc_info=True)
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
                'accuracy': float(acc),
                'f1': float(f1),
                'best_model': best_model.name,
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'timestamp': str(datetime.now())
            }
            st.session_state.training_complete = True
            st.session_state.step = 3
            logger.info(f"Training completed - Best model: {best_model.name}, Accuracy: {acc:.4f}")
            st.rerun()
    except Exception as e:
        error_msg = f"Training Error: {e}\n{traceback.format_exc()}"
        st.session_state.error_msg = error_msg
        logger.error(error_msg)
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

            st.markdown("### Model Comparison Summary")
            st.markdown("```")
            st.markdown(f"{'Model': <30} {'Accuracy': <10} {'F1-Score': <10} {'Score': <10}")
            st.markdown("-" * 65)
            for entry in sorted(history, key=lambda x: x['final_score'], reverse=True):
                st.markdown(f"{entry['model']: <30} {entry['metrics']['accuracy']: <10.4f} {entry['metrics']['f1']: <10.4f} {entry['final_score']: <10.4f}")
            st.markdown("```")

    st.info("Results and artifacts saved to `outputs/`")
    if st.button("Start New Analysis", type="primary"):
        reset_session()
        st.rerun()

if __name__ == "__main__":
    main()