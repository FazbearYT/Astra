import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import time
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())

script_dir = Path(__file__).parent
if (script_dir / "src").exists():
    sys.path.insert(0, str(script_dir / "src"))
else:
    sys.path.insert(0, str(script_dir))

from model_profiler import DataProfiler
from model_selector import AdaptiveModelSelector, SpecializedModel
from pipeline_config import PipelineConfig, get_default_config, get_fast_config, get_accurate_config, interactive_config
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class DataManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.tabular_dir = data_dir / "tabular"
        self.tabular_dir.mkdir(parents=True, exist_ok=True)

    def create_iris_dataset(self) -> Path:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        filepath = self.tabular_dir / "iris.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def create_wine_dataset(self) -> Path:
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        df['target_name'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
        filepath = self.tabular_dir / "wine.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def create_digits_dataset(self) -> Path:
        from sklearn.datasets import load_digits
        digits = load_digits()
        df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
        df['target'] = digits.target
        filepath = self.tabular_dir / "digits.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def create_titanic_dataset(self) -> Path:
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
        filepath = self.tabular_dir / "titanic.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def create_synthetic_dataset(self) -> Path:
        from sklearn.datasets import make_blobs
        np.random.seed(42)
        X, y = make_blobs(n_samples=500, n_features=5, centers=3, cluster_std=1.5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        df['target_name'] = df['target'].map({0: 'class_A', 1: 'class_B', 2: 'class_C'})
        filepath = self.tabular_dir / "synthetic.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def create_large_dataset(self) -> Path:
        from sklearn.datasets import make_classification
        print("\nГенерация большого датасета (100,000 строк)... ")
        X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=5, n_clusters_per_class=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        filepath = self.tabular_dir / "large_test.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        return filepath

    def auto_detect_datasets(self) -> List[Dict]:
        available = []
        if self.tabular_dir.exists():
            for csv_file in self.tabular_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    available.append({'name': csv_file.stem, 'path': csv_file, 'rows': len(df), 'cols': len(df.columns)})
                except Exception:
                    pass
        return available

    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def create_all_test_datasets(self) -> List[Path]:
        paths = []
        paths.append(self.create_iris_dataset())
        paths.append(self.create_wine_dataset())
        paths.append(self.create_digits_dataset())
        paths.append(self.create_titanic_dataset())
        paths.append(self.create_synthetic_dataset())
        return paths

class AdaptiveMLApp:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.target_column = None
        self.feature_columns = None
        self.profile = None
        self.selector: Optional[AdaptiveModelSelector] = None
        self.best_model: Optional[SpecializedModel] = None
        self.results = {}
        self.output_dir = None
        self.session_id = None
        self.pipeline_config: Optional[PipelineConfig] = None
        self.data_manager = None

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        self.clear_screen()
        print(title.upper())
        print("-" * 40)

    def print_step(self, step_num: int, text: str):
        print(f"\nШаг {step_num}: {text} ")
        print("-" * 40)

    def get_user_choice(self, prompt: str, options: list) -> int:
        print(f"\n{prompt} ")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option} ")
        print(f"  0. Выход ")
        while True:
            try:
                choice = int(input(f"\nВаш выбор (0-{len(options)}):  "))
                if choice == 0:
                    print("\nПрограмма завершена ")
                    sys.exit(0)
                elif 1 <= choice <= len(options):
                    return choice
                else:
                    print(f"Введите число от 0 до {len(options)} ")
            except ValueError:
                print("Введите корректное число ")
            except KeyboardInterrupt:
                print("\n\nПрограмма завершена ")
                sys.exit(0)

    def setup_output_directory(self):
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        existing_runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if existing_runs:
            last_num = int(existing_runs[-1].name.split("_")[1])
            self.session_id = last_num + 1
        else:
            self.session_id = 1
        self.output_dir = outputs_dir / f"run_{self.session_id:03d}"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        print(f"Результаты: {self.output_dir} ")
        return self.output_dir

    def initialize_data_manager(self):
        data_dir = Path("data")
        self.data_manager = DataManager(data_dir)

    def show_main_menu(self) -> int:
        self.print_header("Adaptive ML System ")
        print("Автоматический выбор ML моделей\n ")
        print("МЕНЮ: ")
        return self.get_user_choice("\nВыберите действие ", ["Запустить анализ данных ", "Создать тестовые датасеты ", "Просмотреть результатов "])

    def run_data_analysis(self):
        try:
            self.setup_output_directory()
            if not self.configure_pipeline(): return False
            if not self.load_data(): return False
            if not self.select_target_column(): return False
            if not self.analyze_data(): return False
            if not self.select_and_train_model(): return False
            if not self.evaluate_and_show_results(): return False
            if not self.predict_new_data(): return False
            self.show_summary()
            return True
        except Exception as e:
            print(f"\nОшибка: {e} ")
            import traceback
            traceback.print_exc()
            return False

    def create_test_datasets_menu(self):
        self.print_header("Создание тестовых датасетов ")
        print("Выберите датасет: ")
        print("  1. Iris (150 строк) ")
        print("  2. Wine (178 строк) ")
        print("  3. Digits (1797 строк) ")
        print("  4. Titanic (891 строк) ")
        print("  5. Synthetic (500 строк) ")
        print("  6. Все датасеты ")
        print("  7. Большой датасет (100K строк) ")
        print("  0. Назад ")
        choice = self.get_user_choice("\nВаш выбор ", ["Iris ", "Wine ", "Digits ", "Titanic ", "Synthetic ", "Все датасеты ", "Большой датасет "])
        if choice == 1: path = self.data_manager.create_iris_dataset(); print(f"Создан: {path} ")
        elif choice == 2: path = self.data_manager.create_wine_dataset(); print(f"Создан: {path} ")
        elif choice == 3: path = self.data_manager.create_digits_dataset(); print(f"Создан: {path} ")
        elif choice == 4: path = self.data_manager.create_titanic_dataset(); print(f"Создан: {path} ")
        elif choice == 5: path = self.data_manager.create_synthetic_dataset(); print(f"Создан: {path} ")
        elif choice == 6: paths = self.data_manager.create_all_test_datasets(); print(f"Создано {len(paths)} датасетов ")
        elif choice == 7: path = self.data_manager.create_large_dataset(); print(f"Создан: {path} ")
        input("\nНажмите Enter для продолжения... ")

    def view_results_menu(self):
        self.print_header("Просмотр результатов ")
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            print("Нет результатов ")
            input("\nНажмите Enter для продолжения... "); return
        runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not runs:
            print("Нет результатов ")
            input("\nНажмите Enter для продолжения... "); return
        print(f"Найдено сессий: {len(runs)} ")
        print("\nПоследние сессии: ")
        for run in runs[-5:]:
            results_file = run / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    print(f"  {run.name}: {results.get('best_model', 'N/A')} (Accuracy: {results.get('accuracy', 0):.4f}) ")
                except json.JSONDecodeError:
                    print(f"  {run.name}: (Ошибка чтения results.json) ")
            else:
                print(f"  {run.name}: (results.json не найден) ")
        input("\nНажмите Enter для продолжения... ")

    def configure_pipeline(self):
        self.print_step(0, "Настройка Pipeline ")
        print("Режим конфигурации: ")
        print("  1. Быстрая настройка ")
        print("  2. Расширенная настройка (интерактивно) ")
        print("  3. Стандартная (по умолчанию) ")
        mode_options = ["Быстрая настройка ", "Расширенная настройка ", "Стандартная (по умолчанию) "]
        mode = self.get_user_choice("\nВыберите режим ", mode_options)
        if mode == 1:
            print("\nПрофиль: ")
            print("  1. Быстрый (2 модели, 3 CV) ")
            print("  2. Стандартный (5 моделей, 5 CV) ")
            print("  3. Точный (5 моделей, 10 CV) ")
            profile_options = ["Быстрый ", "Стандартный ", "Точный "]
            profile = self.get_user_choice("\nПрофиль ", profile_options)
            if profile == 1: self.pipeline_config = get_fast_config()
            elif profile == 2: self.pipeline_config = get_default_config()
            else: self.pipeline_config = get_accurate_config()
            print(f"\nМоделей: {sum(1 for m in self.pipeline_config.models.values() if m.enabled)} ")
            print(f"CV folds: {self.pipeline_config.training.get('cv_folds', 'N/A')} ")
        elif mode == 2:
            self.pipeline_config = interactive_config()
        else:
            self.pipeline_config = get_default_config()
        config_path = self.output_dir / "pipeline_config.json"
        try:
            self.pipeline_config.save(str(config_path))
            print(f"Конфигурация сохранена: {config_path} ")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e} ")
        return True

    def load_data(self):
        self.print_step(1, "Загрузка данных ")
        available = self.data_manager.auto_detect_datasets()
        if available:
            print("\nНайдены датасеты: ")
            options_display = [f"{ds['name']} ({ds['rows']} строк)" for ds in available]
            choice = self.get_user_choice("\nВыберите датасет ", options_display + ["Создать тестовые датасеты "])
            if choice <= len(available):
                selected = available[choice - 1]
                self.data = self.data_manager.load_dataset(selected['path'])
                print(f"Загружено {len(self.data)} строк из {selected['name']} ")
                return True
            elif choice == len(available) + 1:
                self.create_test_datasets_menu()
                return self.load_data()
            return False
        else:
            print("Датасеты не найдены. ")
            choice = self.get_user_choice("\nЧто делать ", ["Создать тестовые датасеты ", "Назад в главное меню "])
            if choice == 1:
                self.create_test_datasets_menu()
                return self.load_data()
            else:
                return False

    def select_target_column(self):
        self.print_step(2, "Целевая переменная ")
        target_candidates = ['target', 'class', 'label', 'category', 'target_name', 'flower_name', 'species']
        found_target = False
        for col in target_candidates:
            if col in self.data.columns:
                self.target_column = col
                print(f"Автоматически определена целевая колонка: {col} ")
                found_target = True
                break
        if not found_target:
            columns = list(self.data.columns)
            print("Доступные колонки: ")
            for i, col in enumerate(columns, 1):
                sample = self.data[col].iloc[0] if len(self.data) > 0 else "N/A"
                unique_vals = self.data[col].nunique()
                print(f"  {i:2}. {col:30} | тип: {str(self.data[col].dtype):10} | Уникальных: {unique_vals} ")
            choice = self.get_user_choice("\nКакую колонку предсказываем ", columns)
            self.target_column = columns[choice - 1]
        self.y = self.data[self.target_column].values
        all_feature_cols = [col for col in self.data.columns if col != self.target_column]
        self.feature_columns = [col for col in all_feature_cols if pd.api.types.is_numeric_dtype(self.data[col])]
        if not self.feature_columns:
            print("Ошибка: В датасете нет числовых колонок для использования в качестве признаков! ")
            return False
        self.X = self.data[self.feature_columns].values
        print(f"Признаков: {len(self.feature_columns)} ")
        print(f"Образцов: {self.X.shape[0]} ")
        unique_classes = np.unique(self.y)
        print(f"Классов: {len(unique_classes)} ")
        if len(unique_classes) < 2:
            print("Предупреждение: Целевая переменная содержит менее 2 уникальных классов. Это может вызвать проблемы с обучением. ")
            return False
        return True

    def analyze_data(self):
        self.print_step(3, "Анализ данных ")
        profiler = DataProfiler(dataset_name="User_Dataset")
        self.profile = profiler.profile_tabular_data(self.X, self.y, feature_names=self.feature_columns)
        profiler.print_summary()
        profile_path = self.output_dir / "profile.json"
        try:
            self.profile.save(str(profile_path))
            print(f"Профиль данных сохранен: {profile_path} ")
        except Exception as e:
            print(f"Ошибка сохранения профиля данных: {e} ")
        try:
            viz_path = self.output_dir / "visualizations" / "data_profile.png"
            profiler.visualize_profile(save_path=str(viz_path))
            plt.close('all')
            print(f"График профиля данных: {viz_path} ")
        except Exception as e:
            print(f"Ошибка визуализации профиля данных: {e} ")
        return True

    def select_and_train_model(self):
        self.print_step(4, "Обучение моделей ")
        self.selector = AdaptiveModelSelector()
        if self.pipeline_config:
            print("Конфигурация Pipeline: ")
            enabled_models = [m.name for m in self.pipeline_config.models.values() if m.enabled]
            print(f"  Включенные модели: {', '.join(enabled_models) if enabled_models else 'Нет'} ")
            print(f"  CV folds: {self.pipeline_config.training.get('cv_folds', 'N/A')} ")
            print(f"  Вес Accuracy: {self.pipeline_config.scoring.get('accuracy_weight', 'N/A')} ")
            print(f"  Вес F1-Score: {self.pipeline_config.scoring.get('f1_weight', 'N/A')} ")
            print(f"  Вес CV Score: {self.pipeline_config.scoring.get('cv_weight', 'N/A')} ")
            try:
                self.selector.register_models_from_pipeline_config(self.pipeline_config)
            except ValueError as e:
                print(f"Ошибка при регистрации моделей из конфигурации: {e} ")
                return False
        else:
            print("Конфигурация Pipeline не установлена. Используются стандартные параметры. ")
            default_config = get_default_config()
            self.selector.register_models_from_pipeline_config(default_config)
        if not self.selector.models:
            print("Ошибка: Нет моделей для обучения. Проверьте конфигурацию pipeline. ")
            return False
        cv_folds = self.pipeline_config.training.get('cv_folds', 5)
        use_cv = self.pipeline_config.training.get('use_cv_in_scoring', False)
        accuracy_weight = self.pipeline_config.scoring.get('accuracy_weight', 0.7)
        f1_weight = self.pipeline_config.scoring.get('f1_weight', 0.3)
        cv_weight = self.pipeline_config.scoring.get('cv_weight', 0.0)
        self.best_model, self.X_test, self.y_test = self.selector.profile_and_select(
            self.X, self.y, data_profile=self.profile.to_dict(), cv_folds=cv_folds,
            use_cv_in_scoring=use_cv, accuracy_weight=accuracy_weight, f1_weight=f1_weight,
            cv_weight=cv_weight
        )
        return True

    def evaluate_and_show_results(self):
        self.print_step(5, "Результаты ")
        if self.best_model is None:
            print("Ошибка: Лучшая модель не выбрана. ")
            return False

        y_pred = self.best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

        print(f"\nВыбранная модель: {self.best_model.name} ")
        print(f"Итоговая точность на отложенной тестовой выборке: {accuracy:.4f} ({accuracy*100:.2f}%) ")
        print(f"Итоговый F1-Score на отложенной тестовой выборке: {f1:.4f} ")
        print("\nОтчет по классификации: ")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        try:
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=np.unique(self.y), yticklabels=np.unique(self.y))
            ax.set_title(f'Матрица ошибок - {self.best_model.name}')
            ax.set_xlabel('Предсказанный класс')
            ax.set_ylabel('Истинный класс')
            cm_path = self.output_dir / "visualizations" / "confusion_matrix.png"
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            plt.close('all')
            print(f"Матрица ошибок сохранена: {cm_path} ")
        except Exception as e:
            print(f"Ошибка при создании матрицы ошибок: {e} ")
        self.results = {
            'session_id': self.session_id,
            'dataset_name': getattr(self.profile, 'dataset_name', 'unknown'),
            'target_column': self.target_column,
            'best_model': self.best_model.name,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'n_samples': int(self.X.shape[0]),
            'n_features': int(self.X.shape[1]),
            'timestamp': str(datetime.now().isoformat())
        }
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Результаты анализа сохранены: {results_path} ")
        models_dir = self.output_dir / "models"
        self.selector.save_all_models(str(models_dir))
        print(f"Обученные модели сохранены в: {models_dir} ")
        return True

    def predict_new_data(self):
        self.print_step(6, "Предсказание ")
        if self.best_model is None:
            print("Невозможно сделать предсказания, лучшая модель не обучена. ")
            return True
        choice_options = ["Ввести данные вручную ", "Загрузить данные из CSV-файла ", "Завершить "]
        choice = self.get_user_choice("\nДействие ", choice_options)
        if choice == 1:
            print(f"Введите {len(self.feature_columns)} числовых значений, разделенных пробелами (например, {', '.join(self.feature_columns[:3])}...): ")
            try:
                input_str = input("> ").strip()
                values = [float(v) for v in input_str.replace(',', '.').split()]
                if len(values) != len(self.feature_columns):
                    print(f"Ошибка: Ожидалось {len(self.feature_columns)} значений, получено {len(values)}. Попробуйте снова. ")
                    return False
                prediction_input = np.array([values])
                prediction = self.selector.predict(prediction_input)[0]
                prediction_proba = self.selector.predict_proba(prediction_input)
                print(f"\nПредсказанный класс: {prediction} ")
                print(f"Вероятности классов: {prediction_proba} ")
            except ValueError as e:
                print(f"Ошибка ввода: Убедитесь, что вы вводите числа, разделенные пробелами. {e} ")
            except Exception as e:
                print(f"Произошла непредвиденная ошибка при предсказании: {e} ")
        elif choice == 2:
            filepath = input("Введите путь к CSV-файлу с новыми данными: ").strip()
            if Path(filepath).exists():
                try:
                    new_data_df = pd.read_csv(filepath)
                    missing_cols = [col for col in self.feature_columns if col not in new_data_df.columns]
                    if missing_cols:
                        print(f"Ошибка: В новом CSV-файле отсутствуют необходимые колонки признаков: {', '.join(missing_cols)} ")
                        return False
                    new_x_for_prediction = new_data_df[self.feature_columns].values
                    predictions = self.selector.predict(new_x_for_prediction)
                    output_df = new_data_df.copy()
                    output_df['predicted_target'] = predictions
                    if hasattr(self.best_model.model, 'predict_proba'):
                        probabilities = self.best_model.predict_proba(new_x_for_prediction)
                        for i, class_label in enumerate(self.best_model.model.classes_):
                             output_df[f'probability_class_{class_label}'] = probabilities[:, i]
                    output_path = self.output_dir / f"predictions_{Path(filepath).stem}.csv"
                    output_df.to_csv(output_path, index=False, encoding='utf-8')
                    print(f"Предсказания сохранены в: {output_path} ")
                except pd.errors.EmptyDataError:
                    print(f"Ошибка: CSV-файл '{filepath}' пуст. ")
                except Exception as e:
                    print(f"Ошибка при обработке CSV-файла для предсказания: {e} ")
            else:
                print(f"Ошибка: Файл '{filepath}' не найден. ")
        elif choice == 3:
            return True
        return True

    def show_summary(self):
        self.print_header("Анализ завершен! ")
        if self.best_model and self.results:
            print(f"Выбранная модель: {self.best_model.name} ")
            print(f"Итоговая точность: {self.results['accuracy']*100:.2f}% ")
            f1 = self.results.get('f1_score', 0.0)
            print(f"Итоговый F1-Score: {f1*100:.2f}% ")
            print(f"\nВсе результаты и артефакты сохранены в: {self.output_dir} ")
        else:
            print("Не удалось завершить анализ или получить результаты. ")
        input("\nНажмите Enter для выхода... ")

    def run(self):
        self.initialize_data_manager()
        while True:
            choice = self.show_main_menu()
            if choice == 1: self.run_data_analysis()
            elif choice == 2: self.create_test_datasets_menu()
            elif choice == 3: self.view_results_menu()
            elif choice == 0:
                print("\nПрограмма завершена ")
                break

def main():
    app = AdaptiveMLApp()
    app.run()

if __name__ == "__main__":
    main()