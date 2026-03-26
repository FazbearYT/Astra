"""
Adaptive ML System - Единая точка входа
"""

import sys
import os
from pathlib import Path
import json
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import time

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_profiler import DataProfiler
from model_selector import AdaptiveModelSelector
from pipeline_config import PipelineConfig, get_default_config, get_fast_config, get_accurate_config, interactive_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        df['target'] = ((df['gender'] == 1) & (df['pclass'] < 3) |
                       (df['age'] < 10)).astype(int)

        filepath = self.tabular_dir / "titanic.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')

        return filepath

    def create_synthetic_dataset(self) -> Path:
        from sklearn.datasets import make_blobs

        np.random.seed(42)
        X, y = make_blobs(n_samples=500, n_features=5, centers=3,
                         cluster_std=1.5, random_state=42)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        df['target_name'] = df['target'].map({0: 'class_A', 1: 'class_B', 2: 'class_C'})

        filepath = self.tabular_dir / "synthetic.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')

        return filepath

    def create_large_dataset(self) -> Path:
        from sklearn.datasets import make_classification

        print("\nГенерация большого датасета (100,000 строк)...")
        print("Это может занять 1-2 минуты")

        X, y = make_classification(
            n_samples=100000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            n_clusters_per_class=2,
            random_state=42
        )

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
                    available.append({
                        'name': csv_file.stem,
                        'path': csv_file,
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
                except Exception:
                    pass

        return available

    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def create_all_test_datasets(self) -> List[Path]:
        print("\nСоздание всех тестовых датасетов...")

        paths = []
        paths.append(self.create_iris_dataset())
        paths.append(self.create_wine_dataset())
        paths.append(self.create_digits_dataset())
        paths.append(self.create_titanic_dataset())
        paths.append(self.create_synthetic_dataset())

        print(f"\nСоздано {len(paths)} датасетов в {self.tabular_dir}")

        return paths


class AdaptiveMLApp:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.target_column = None
        self.feature_columns = None
        self.profile = None
        self.selector = None
        self.best_model = None
        self.results = {}
        self.output_dir = None
        self.session_id = None
        self.pipeline_config = None
        self.data_manager = None

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        self.clear_screen()
        print("="*70)
        print(title.upper())
        print("="*70)
        print()

    def print_step(self, step_num: int, text: str):
        print(f"\n{'='*70}")
        print(f"ШАГ {step_num}: {text}")
        print("="*70)

    def get_user_choice(self, prompt: str, options: list) -> int:
        print(f"\n{prompt}")
        print("-" * 50)

        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

        print(f"  0. Выход")

        while True:
            try:
                choice = int(input(f"\nВаш выбор (0-{len(options)}): "))
                if choice == 0:
                    print("\nПрограмма завершена")
                    sys.exit(0)
                elif 1 <= choice <= len(options):
                    return choice
                else:
                    print(f"Введите число от 0 до {len(options)}")
            except ValueError:
                print("Введите корректное число")
            except KeyboardInterrupt:
                print("\n\nПрограмма завершена")
                sys.exit(0)

    def setup_output_directory(self):
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)

        existing_runs = sorted([d for d in outputs_dir.iterdir()
                               if d.is_dir() and d.name.startswith("run_")])

        if existing_runs:
            last_num = int(existing_runs[-1].name.split("_")[1])
            self.session_id = last_num + 1
        else:
            self.session_id = 1

        self.output_dir = outputs_dir / f"run_{self.session_id:03d}"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        print(f"\nРезультаты: {self.output_dir}")
        return self.output_dir

    def initialize_data_manager(self):
        data_dir = Path("data")
        self.data_manager = DataManager(data_dir)

    def show_main_menu(self) -> int:
        self.print_header("Adaptive ML System")

        print("Автоматический выбор ML моделей")
        print()
        print("МЕНЮ:")
        print("  1. Запустить анализ данных")
        print("  2. Создать тестовые датасеты")
        print("  3. Просмотреть результаты")
        print("  0. Выход")

        return self.get_user_choice("\nВыберите действие", [
            "Запустить анализ данных",
            "Создать тестовые датасеты",
            "Просмотреть результаты"
        ])

    def run_data_analysis(self):
        try:
            self.setup_output_directory()

            if not self.configure_pipeline():
                return False
            if not self.load_data():
                return False
            if not self.select_target_column():
                return False
            if not self.analyze_data():
                return False
            if not self.select_and_train_model():
                return False
            if not self.evaluate_and_show_results():
                return False
            if not self.predict_new_data():
                return False

            self.show_summary()
            return True

        except Exception as e:
            print(f"\nОшибка: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_test_datasets_menu(self):
        self.print_header("Создание тестовых датасетов")

        print("\nВыберите датасет для создания:")
        print("  1. Iris (150 строк, 3 класса)")
        print("  2. Wine (178 строк, 3 класса)")
        print("  3. Digits (1797 строк, 10 классов)")
        print("  4. Titanic (891 строк, 2 класса)")
        print("  5. Synthetic (500 строк, 3 класса)")
        print("  6. Все датасеты сразу")
        print("  7. Большой датасет (100,000 строк)")
        print("  0. Назад")

        choice = self.get_user_choice("\nВаш выбор", [
            "Iris",
            "Wine",
            "Digits",
            "Titanic",
            "Synthetic",
            "Все датасеты",
            "Большой датасет"
        ])

        if choice == 1:
            path = self.data_manager.create_iris_dataset()
            print(f"\nСоздан: {path}")
        elif choice == 2:
            path = self.data_manager.create_wine_dataset()
            print(f"\nСоздан: {path}")
        elif choice == 3:
            path = self.data_manager.create_digits_dataset()
            print(f"\nСоздан: {path}")
        elif choice == 4:
            path = self.data_manager.create_titanic_dataset()
            print(f"\nСоздан: {path}")
        elif choice == 5:
            path = self.data_manager.create_synthetic_dataset()
            print(f"\nСоздан: {path}")
        elif choice == 6:
            paths = self.data_manager.create_all_test_datasets()
            print(f"\nСоздано {len(paths)} датасетов")
        elif choice == 7:
            path = self.data_manager.create_large_dataset()
            print(f"\nСоздан: {path}")

        input("\nНажмите Enter для продолжения...")

    def view_results_menu(self):
        self.print_header("Просмотр результатов")

        outputs_dir = Path("outputs")

        if not outputs_dir.exists():
            print("\nНет результатов")
            input("\nНажмите Enter для продолжения...")
            return

        runs = sorted([d for d in outputs_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run_")])

        if not runs:
            print("\nНет результатов")
            input("\nНажмите Enter для продолжения...")
            return

        print(f"\nНайдено сессий: {len(runs)}")
        print("\nПоследние 5 сессий:")

        for run in runs[-5:]:
            results_file = run / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"  {run.name}: {results['best_model']} (Accuracy: {results['accuracy']:.4f})")

        print(f"\nПоследняя сессия: outputs/latest/")

        input("\nНажмите Enter для продолжения...")

    def configure_pipeline(self):
        self.print_step(0, "Настройка Pipeline")

        print("\nРежимы конфигурации:")
        print("  1. Быстрая настройка (стандартные параметры)")
        print("  2. Расширенная настройка (все параметры)")
        print("  3. Пропустить (использовать значения по умолчанию)")

        mode = self.get_user_choice("\nВыберите режим", [
            "Быстрая настройка",
            "Расширенная настройка",
            "Пропустить"
        ])

        if mode == 1:
            print("\nВыберите профиль:")
            print("  1. Быстрый (2 модели, 3 CV folds) - для маленьких датасетов")
            print("  2. Стандартный (5 моделей, 5 CV folds) - рекомендуется")
            print("  3. Точный (5 моделей, 10 CV folds) - для важных задач")

            profile = self.get_user_choice("\nПрофиль", [
                "Быстрый",
                "Стандартный",
                "Точный"
            ])

            if profile == 1:
                self.pipeline_config = get_fast_config()
            elif profile == 2:
                self.pipeline_config = get_default_config()
            else:
                self.pipeline_config = get_accurate_config()

            print("\nОсновные параметры:")
            print(f"  - Моделей: {sum(1 for m in self.pipeline_config.models.values() if m.enabled)}")
            print(f"  - CV folds: {self.pipeline_config.training['cv_folds']}")
            print(f"  - Accuracy вес: {self.pipeline_config.scoring['accuracy_weight']}")
            print(f"  - F1-Score вес: {self.pipeline_config.scoring['f1_weight']}")

        elif mode == 2:
            self.pipeline_config = interactive_config()
            print("\nРасширенная конфигурация применена")
        else:
            self.pipeline_config = get_default_config()
            print("\nИспользуется конфигурация по умолчанию")

        config_path = self.output_dir / "pipeline_config.json"
        self.pipeline_config.save(str(config_path))

        input("\nНажмите Enter для продолжения...")
        return True

    def load_data(self):
        self.print_step(1, "Загрузка данных")

        available = self.data_manager.auto_detect_datasets()

        if available:
            print("\nНАЙДЕНЫ ДАТАСЕТЫ:")
            print("="*70)

            options = []
            for i, ds in enumerate(available, 1):
                print(f"  {i}. {ds['name']} ({ds['rows']} строк, {ds['cols']} колонок)")
                options.append(ds)

            options.append({'type': 'create_new', 'name': 'Создать тестовые датасеты'})
            print(f"  {len(options)}. Создать тестовые датасеты")

            choice = self.get_user_choice("\nВыберите датасет",
                                         [opt['name'] for opt in options])

            if choice <= len(available):
                selected = options[choice - 1]
                self.data = self.data_manager.load_dataset(selected['path'])
                print(f"\nЗагружено {len(self.data)} строк")
                return True
            elif choice == len(options):
                self.create_test_datasets_menu()
                return self.load_data()

            return False

        else:
            print("\nДатасеты не найдены!")

            choice = self.get_user_choice("\nЧто делать", [
                "Создать тестовые датасеты",
                "Назад в главное меню"
            ])

            if choice == 1:
                self.create_test_datasets_menu()
                return self.load_data()
            else:
                return False

    def select_target_column(self):
        self.print_step(2, "Целевая переменная")

        target_candidates = ['target', 'class', 'label', 'category',
                           'target_name', 'flower_name', 'species']

        for col in target_candidates:
            if col in self.data.columns:
                self.target_column = col
                print(f"\nНайдена target колонка: {col}")
                break

        if self.target_column is None:
            print("\nКолонки:")
            columns = list(self.data.columns)

            for i, col in enumerate(columns, 1):
                sample = self.data[col].iloc[0] if len(self.data) > 0 else "N/A"
                unique_vals = self.data[col].nunique()
                print(f"  {i:2}. {col:30} | тип: {str(self.data[col].dtype):10} | уникальных: {unique_vals:4}")

            choice = self.get_user_choice("\nКакую колонку предсказываем", columns)
            self.target_column = columns[choice - 1]

        self.y = self.data[self.target_column].values

        all_feature_cols = [col for col in self.data.columns if col != self.target_column]

        self.feature_columns = []
        for col in all_feature_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.feature_columns.append(col)

        if len(self.feature_columns) == 0:
            print("\nНет числовых колонок!")
            return False

        self.X = self.data[self.feature_columns].values

        print(f"\nПризнаков: {len(self.feature_columns)}")
        print(f"Образцов: {self.X.shape[0]}")
        print(f"Классов: {len(np.unique(self.y))}")

        self.explain_complexity_calculation(self.X.shape[0], self.X.shape[1])

        return True

    def explain_complexity_calculation(self, n_samples: int, n_features: int):
        ratio = n_samples / n_features if n_features > 0 else 0

        print("\n" + "="*70)
        print("КАК ОПРЕДЕЛЯЕТСЯ СЛОЖНОСТЬ ДАТАСЕТА")
        print("="*70)
        print(f"\nФормула: отношение образцов к признакам")
        print(f"  - Образцов: {n_samples}")
        print(f"  - Признаков: {n_features}")
        print(f"  - Отношение: {ratio:.1f}")
        print("\nКритерии:")
        print("  - Простой: отношение < 50")
        print("  - Средний: 50 <= отношение < 200")
        print("  - Сложный: отношение >= 200")

        if ratio < 50:
            complexity = "ПРОСТОЙ"
            recommendation = "Подойдут простые модели: Random Forest, SVM, Logistic Regression"
        elif ratio < 200:
            complexity = "СРЕДНЕЙ СЛОЖНОСТИ"
            recommendation = "Рекомендуются: Gradient Boosting, Random Forest, Neural Network"
        else:
            complexity = "СЛОЖНЫЙ"
            recommendation = "Лучше всего: Neural Network, Gradient Boosting"

        print(f"\nВаш датасет: {complexity}")
        print(f"\nРекомендация: {recommendation}")
        print("="*70)

        input("\nНажмите Enter для продолжения...")

    def analyze_data(self):
        self.print_step(3, "Анализ данных")

        print("\nАнализ характеристик датасета...")
        print("  >>> Вычисление статистик признаков")

        profiler = DataProfiler(dataset_name="User_Dataset")
        self.profile = profiler.profile_tabular_data(
            self.X, self.y,
            feature_names=self.feature_columns
        )

        print("  >>> Анализ распределения классов")
        print("  >>> Поиск выбросов и аномалий")
        print("  >>> Оценка корреляций")
        print("  >>> Формирование рекомендаций")

        profiler.print_summary()

        profile_path = self.output_dir / "profile.json"
        self.profile.save(str(profile_path))

        try:
            viz_path = self.output_dir / "visualizations" / "data_profile.png"
            profiler.visualize_profile(save_path=str(viz_path))
            plt.close('all')
            print(f"\nГрафик сохранен: {viz_path}")
        except Exception as e:
            print(f"\nОшибка визуализации: {e}")

        input("\nНажмите Enter для продолжения...")
        return True

    def select_and_train_model(self):
        self.print_step(4, "Обучение моделей")

        print("\nИнициализация селектора моделей...")
        self.selector = AdaptiveModelSelector()

        if self.pipeline_config:
            print("\nКонфигурация pipeline:")
            enabled_models = [m.name for m in self.pipeline_config.models.values() if m.enabled]
            print(f"  - Модели: {', '.join(enabled_models)}")
            print(f"  - CV folds: {self.pipeline_config.training['cv_folds']}")
            print(f"  - Accuracy вес: {self.pipeline_config.scoring['accuracy_weight']}")
            print(f"  - F1-Score вес: {self.pipeline_config.scoring['f1_weight']}")

            print("  >>> Создание моделей")
            self.selector.create_default_models(
                model_types=[k for k, v in self.pipeline_config.models.items() if v.enabled]
            )
        else:
            print("  >>> Создание моделей по умолчанию")
            self.selector.create_default_models()

        print("\nПодготовка данных...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\nРазделение данных:")
        print(f"  - Train: {len(X_train)} образцов ({len(X_train)/len(self.X)*100:.0f}%)")
        print(f"  - Test: {len(X_test)} образцов ({len(X_test)/len(self.X)*100:.0f}%)")

        cv_folds = self.pipeline_config.training['cv_folds'] if self.pipeline_config else 5
        use_cv = self.pipeline_config.training['use_cv_in_scoring'] if self.pipeline_config else False

        print(f"\nПараметры обучения:")
        print(f"  - Кросс-валидация: {cv_folds} folds")
        print(f"  - CV в scoring: {'Да' if use_cv else 'Нет'}")
        print(f"  - Метрики: Accuracy (70%) + F1-Score (30%)")

        print("  >>> Начало обучения и тестирования моделей")
        print("\n" + "="*70)

        self.best_model = self.selector.profile_and_select(
            self.X, self.y,
            data_profile=self.profile.to_dict(),
            cv_folds=cv_folds,
            use_cv_in_scoring=use_cv
        )

        input("\nНажмите Enter для продолжения...")
        return True

    def evaluate_and_show_results(self):
        self.print_step(5, "Результаты")

        print("\nФинальная оценка на тестовой выборке...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print("  >>> Генерация предсказаний")
        y_pred = self.selector.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "="*70)
        print("ИТОГИ")
        print("="*70)
        print(f"\nМодель: {self.best_model.name}")
        print(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nОтчет:")
        print(classification_report(y_test, y_pred))

        try:
            print("  >>> Построение матрицы ошибок")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Matrix - {self.best_model.name}')
            cm_path = self.output_dir / "visualizations" / "confusion_matrix.png"
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            plt.close('all')
            print(f"\nМатрица ошибок: {cm_path}")
        except Exception as e:
            print(f"\nОшибка: {e}")

        self.results = {
            'session_id': self.session_id,
            'target_column': self.target_column,
            'best_model': self.best_model.name,
            'accuracy': float(accuracy),
            'n_samples': int(self.X.shape[0]),
            'n_features': int(self.X.shape[1]),
            'timestamp': str(datetime.now())
        }

        print("  >>> Сохранение результатов")
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты: {results_path}")

        models_dir = self.output_dir / "models"
        print("  >>> Сохранение моделей")
        self.selector.save_all_models(str(models_dir))
        print(f"Модели: {models_dir}")

        input("\nНажмите Enter для продолжения...")
        return True

    def predict_new_data(self):
        self.print_step(6, "Предсказание")

        choice = self.get_user_choice("\nДействие", [
            "Ввести данные",
            "Загрузить CSV",
            "Завершить"
        ])

        if choice == 1:
            print(f"\nВведите {len(self.feature_columns)} значений:")
            try:
                values = [float(v) for v in input("> ").strip().replace(',', ' ').split()]
                if len(values) != len(self.feature_columns):
                    print(f"Ожидалось {len(self.feature_columns)}")
                    return False

                prediction = self.selector.predict(np.array([values]))[0]
                print(f"\nКласс: {prediction}")
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == 2:
            filepath = input("\nCSV путь: ").strip()
            if Path(filepath).exists():
                try:
                    new_data = pd.read_csv(filepath)
                    new_x = new_data[self.feature_columns].values
                    predictions = self.selector.predict(new_x)
                    new_data['prediction'] = predictions
                    output_path = self.output_dir / "predictions.csv"
                    new_data.to_csv(output_path, index=False)
                    print(f"\nПредсказания: {output_path}")
                except Exception as e:
                    print(f"Ошибка: {e}")

        return True

    def show_summary(self):
        self.print_header("Готово!")
        print("Успешно!")
        print(f"\n{self.output_dir}")
        print(f"Модель: {self.best_model.name}")
        print(f"Accuracy: {self.results['accuracy']*100:.2f}%")
        print(f"\noutputs/latest/")
        print("\nСпасибо!")

    def run(self):
        self.initialize_data_manager()

        while True:
            choice = self.show_main_menu()

            if choice == 1:
                self.run_data_analysis()
                input("\nНажмите Enter для продолжения...")
            elif choice == 2:
                self.create_test_datasets_menu()
            elif choice == 3:
                self.view_results_menu()
            elif choice == 0:
                print("\nПрограмма завершена")
                break


def main():
    app = AdaptiveMLApp()
    app.run()


if __name__ == "__main__":
    main()