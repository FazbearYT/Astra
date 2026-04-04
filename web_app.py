import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.model_profiler import DataProfiler
from src.model_selector import AdaptiveModelSelector
from src.pipeline_config import get_default_config, get_fast_config, get_accurate_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Astra ML System", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")

if 'step' not in st.session_state: st.session_state.step = 0
if 'data' not in st.session_state: st.session_state.data = None
if 'results' not in st.session_state: st.session_state.results = None
if 'profile' not in st.session_state: st.session_state.profile = None
if 'best_model' not in st.session_state: st.session_state.best_model = None
if 'pipeline_config' not in st.session_state: st.session_state.pipeline_config = None
if 'training_complete' not in st.session_state: st.session_state.training_complete = False

def init_directories():
    Path("data/tabular").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

@st.cache_data
def get_available_datasets():
    tabular_dir = Path("data/tabular")
    datasets = []
    if tabular_dir.exists():
        for csv_file in tabular_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                datasets.append({'name': csv_file.stem, 'path': csv_file, 'rows': len(df), 'cols': len(df.columns)})
            except: pass
    return datasets

@st.cache_data
def create_test_dataset_cached(name: str) -> str:
    from sklearn.datasets import load_iris, load_wine, load_digits, make_blobs, make_classification
    tabular_dir = Path("data/tabular")
    if name == "Iris":
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        filepath = tabular_dir / "iris.csv"
    elif name == "Wine":
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        filepath = tabular_dir / "wine.csv"
    elif name == "Digits":
        digits = load_digits()
        df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
        df['target'] = digits.target
        filepath = tabular_dir / "digits.csv"
    elif name == "Titanic":
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({'pclass': np.random.randint(1, 4, n), 'gender': np.random.randint(0, 2, n), 'age': np.random.normal(30, 14, n).clip(0, 80), 'sibsp': np.random.poisson(0.5, n), 'parch': np.random.poisson(0.4, n), 'fare': np.random.exponential(30, n)})
        df['target'] = ((df['gender'] == 1) & (df['pclass'] < 3) | (df['age'] < 10)).astype(int)
        filepath = tabular_dir / "titanic.csv"
    elif name == "Synthetic":
        X, y = make_blobs(n_samples=500, n_features=5, centers=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        filepath = tabular_dir / "synthetic.csv"
    elif name == "Large":
        X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        filepath = tabular_dir / "large_test.csv"
    else: return None
    df.to_csv(filepath, index=False, encoding='utf-8')
    return str(filepath)

def reset_session():
    st.session_state.step = 0
    st.session_state.data = None
    st.session_state.results = None
    st.session_state.profile = None
    st.session_state.best_model = None
    st.session_state.pipeline_config = None
    st.session_state.training_complete = False

def main():
    st.title("Astra ML System")
    st.markdown("Автоматический выбор ML моделей на основе профилирования данных")
    init_directories()
    if st.session_state.step == 0: step_dataset()
    elif st.session_state.step == 1: step_pipeline()
    elif st.session_state.step == 2: step_training()
    elif st.session_state.step == 3: step_results()

def step_dataset():
    st.header("Шаг 1: Загрузка данных")
    st.progress(0.25)
    tab1, tab2, tab3 = st.tabs(["Тестовые датасеты", "Существующие датасеты", "Загрузить свой CSV"])
    with tab1:
        st.subheader("Создать тестовый датасет")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Iris (150 строк)", use_container_width=True):
                path = create_test_dataset_cached("Iris")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
        with col2:
            if st.button("Wine (178 строк)", use_container_width=True):
                path = create_test_dataset_cached("Wine")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
        with col3:
            if st.button("Digits (1797 строк)", use_container_width=True):
                path = create_test_dataset_cached("Digits")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
        col4, col5, col6 = st.columns(3)
        with col4:
            if st.button("Titanic (891 строк)", use_container_width=True):
                path = create_test_dataset_cached("Titanic")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
        with col5:
            if st.button("Synthetic (500 строк)", use_container_width=True):
                path = create_test_dataset_cached("Synthetic")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
        with col6:
            if st.button("Large (100K строк)", use_container_width=True):
                with st.spinner("Генерация большого датасета..."):
                    path = create_test_dataset_cached("Large")
                st.success(f"Создан: {Path(path).name}"); st.session_state.data = pd.read_csv(path)
    with tab2:
        st.subheader("Выбрать существующий датасет")
        datasets = get_available_datasets()
        if datasets:
            dataset_options = {f"{ds['name']} ({ds['rows']} строк, {ds['cols']} колонок)": ds for ds in datasets}
            selected = st.selectbox("Выберите датасет: ", list(dataset_options.keys()))
            if selected:
                ds_info = dataset_options[selected]
                st.info(f"{ds_info['name']}: {ds_info['rows']} строк, {ds_info['cols']} колонок")
                if st.button("Загрузить датасет"):
                    st.session_state.data = pd.read_csv(ds_info['path']); st.success(f"Загружен: {ds_info['name']}")
        else: st.warning("Нет доступных датасетов. Создайте тестовый или загрузите свой.")
    with tab3:
        st.subheader("Загрузить свой CSV файл")
        uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file); st.session_state.data = data
                st.success(f"Загружено {len(data)} строк, {len(data.columns)} колонок")
            except Exception as e: st.error(f"Ошибка загрузки: {e}")
    if st.session_state.data is not None:
        st.success("Датасет загружен"); st.dataframe(st.session_state.data.head())
        if st.button("Далее", type="primary"):
            st.session_state.step = 1; st.rerun()

def step_pipeline():
    st.header("Шаг 2: Настройка Pipeline")
    st.progress(0.5)
    if st.session_state.data is None:
        st.warning("Сначала загрузите данные")
        if st.button("Назад"): st.session_state.step = 0; st.rerun()
        return
    with st.form("pipeline_config_form"):
        st.subheader("Выберите режим конфигурации")
        mode = st.radio("Режим настройки: ", ["Быстрый (2 модели, 3 CV folds)", "Стандартный (3 модели, 5 CV folds)", "Точный (5 моделей, 10 CV folds)", "Расширенный (полная настройка)"], index=1)
        if mode == "Быстрый (2 модели, 3 CV folds)":
            st.session_state.pipeline_config = get_fast_config(); st.success("Применён быстрый пресет")
            st.json({"models": 2, "cv_folds": 3, "accuracy_weight": 0.6, "f1_weight": 0.4})
        elif mode == "Стандартный (3 модели, 5 CV folds)":
            st.session_state.pipeline_config = get_default_config(); st.success("Применён стандартный пресет")
            st.json({"models": 3, "cv_folds": 5, "accuracy_weight": 0.7, "f1_weight": 0.3})
        elif mode == "Точный (5 моделей, 10 CV folds)":
            st.session_state.pipeline_config = get_accurate_config(); st.success("Применён точный пресет")
            st.json({"models": 5, "cv_folds": 10, "accuracy_weight": 0.5, "f1_weight": 0.4, "cv_weight": 0.1})
        elif mode == "Расширенный (полная настройка)":
            st.info("Расширенная настройка конфигурации")
            config = get_accurate_config()
            st.subheader("Выбор моделей (все 5 доступны)")
            for model_key, model_config in config.models.items():
                enabled = st.checkbox(model_config.name, value=model_config.enabled, key=f"model_{model_key}")
                config.models[model_key].enabled = enabled
            st.subheader("Параметры обучения")
            config.training['cv_folds'] = st.number_input("CV Folds", min_value=3, max_value=50, value=config.training['cv_folds'], step=1)
            config.training['test_size'] = st.slider("Test Size", 0.1, 0.4, config.training['test_size'], 0.05)
            st.subheader("Веса метрик (сумма = 1.0)")
            col1, col2, col3 = st.columns(3)
            with col1: acc_w = st.number_input("Accuracy", 0.0, 1.0, config.scoring['accuracy_weight'], 0.1, key="acc_w")
            with col2: f1_w = st.number_input("F1-Score", 0.0, 1.0, config.scoring['f1_weight'], 0.1, key="f1_w")
            with col3: cv_w = st.number_input("CV Score", 0.0, 1.0, config.scoring['cv_weight'], 0.1, key="cv_w")
            total = acc_w + f1_w + cv_w
            if abs(total - 1.0) > 0.01:
                st.warning(f"Сумма весов: {total:.2f} (должна быть 1.0, будет произведена нормализация)")
                if total > 0: acc_w, f1_w, cv_w = acc_w / total, f1_w / total, cv_w / total
                else: acc_w, f1_w, cv_w = 1/3, 1/3, 1/3
            config.scoring['accuracy_weight'] = acc_w; config.scoring['f1_weight'] = f1_w; config.scoring['cv_weight'] = cv_w
            st.session_state.pipeline_config = config; st.success("Конфигурация сохранена")
        submitted = st.form_submit_button("Применить конфигурацию и далее", type="primary")
        if submitted:
            st.session_state.step = 2; st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Назад"): st.session_state.step = 0; st.rerun()

def step_training():
    st.header("Шаг 3: Обучение моделей")
    st.progress(0.75)
    if st.session_state.data is None:
        st.warning("Сначала загрузите данные")
        if st.button("Назад"): st.session_state.step = 0; st.rerun()
        return
    if st.session_state.pipeline_config is None:
        st.warning("Сначала настройте Pipeline")
        if st.button("Назад"): st.session_state.step = 1; st.rerun()
        return
    if st.session_state.training_complete:
        st.success("Обучение завершено. Перейдите к результатам.")
        if st.button("Показать результаты", type="primary"): st.session_state.step = 3; st.rerun()
        return
    if st.button("Запустить обучение", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            data = st.session_state.data
            config = st.session_state.pipeline_config
            target_candidates = ['target', 'class', 'label', 'category', 'target_name']
            target_column = None
            for col in target_candidates:
                if col in data.columns: target_column = col; break
            if target_column is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1: target_column = numeric_cols[-1]; st.warning(f"Целевая колонка не найдена, используется последняя числовая колонка: {target_column}")
                else: st.error("Не найдена целевая колонка"); return
            y = data[target_column].values
            feature_cols = [col for col in data.columns if col != target_column and pd.api.types.is_numeric_dtype(data[col])]
            X = data[feature_cols].values
            status_text.text("Профилирование данных..."); progress_bar.progress(0.1)
            profiler = DataProfiler(dataset_name="Web_Dataset")
            st.session_state.profile = profiler.profile_tabular_data(X, y, feature_names=feature_cols)
            progress_bar.progress(0.2)
            selector = AdaptiveModelSelector()
            selector.register_models_from_pipeline_config(config)
            enabled_models_count = len([m for m in config.models.values() if m.enabled])
            status_text.text(f"Подготовка к обучению {enabled_models_count} моделей...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.training.get('test_size', 0.2), random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
            with st.spinner("Идет процесс выбора и обучения моделей... Это может занять некоторое время."):
                st.session_state.best_model = selector.profile_and_select(X, y, data_profile=st.session_state.profile.to_dict(), cv_folds=config.training.get('cv_folds', 5), use_cv_in_scoring=config.training.get('use_cv_in_scoring', False), accuracy_weight=config.scoring.get('accuracy_weight', 0.7), f1_weight=config.scoring.get('f1_weight', 0.3), cv_weight=config.scoring.get('cv_weight', 0.0))
            y_pred = st.session_state.best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            st.session_state.results = {'accuracy': float(accuracy), 'f1': float(f1), 'best_model': st.session_state.best_model.name, 'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1]), 'timestamp': str(datetime.now())}
            st.session_state.training_complete = True
            progress_bar.progress(1.0)
            status_text.text("Обучение завершено!")
            st.success(f"Обучение завершено! Лучшая модель: {st.session_state.best_model.name}, Accuracy: {accuracy:.4f}")
            st.session_state.step = 3; st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")
            import traceback
            st.code(traceback.format_exc())
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Назад"): st.session_state.step = 1; st.rerun()

def step_results():
    st.header("Шаг 4: Результаты")
    st.progress(1.0)
    if st.session_state.results is None:
        st.warning("Сначала завершите обучение")
        if st.button("Назад к обучению"): st.session_state.step = 2; st.rerun()
        return
    results = st.session_state.results
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Лучшая модель", results['best_model'])
    with col2: st.metric("Accuracy", f"{results['accuracy']:.4f}")
    with col3: st.metric("F1-Score", f"{results.get('f1', 0.0):.4f}")
    if st.session_state.profile:
        st.subheader("Профиль данных")
        profile_data = st.session_state.profile.to_dict()
        st.json({"n_samples": profile_data['n_samples'], "n_features": profile_data['n_features'], "n_classes": profile_data['n_classes'], "data_complexity": profile_data['data_complexity']})
    st.info("Результаты и артефакты (модели, графики) сохраняются в директорию `outputs/`")
    if st.button("Начать заново"): reset_session(); st.rerun()

if __name__ == "__main__":
    main()