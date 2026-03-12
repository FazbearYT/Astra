import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_profiler import DataProfiler
from model_selector import AdaptiveModelSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def create_quick_test_dataset():
    print("\n📊 Создание тестового датасета Iris...")

    tabular_dir = Path("data/tabular")
    tabular_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    filepath = tabular_dir / "iris_auto.csv"
    df.to_csv(filepath, index=False, encoding='utf-8')

    print(f"   ✅ Создан: {filepath}")
    print(f"   📏 {len(df)} строк, {len(df.columns)} колонок")

    return filepath


class AdaptiveMLApp:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.target_column = None
        self.feature_columns = None  # Только numeric колонки
        self.profile = None
        self.selector = None
        self.best_model = None
        self.results = {}
        self.output_dir = None
        self.session_id = None

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        self.clear_screen()
        print("="*70)
        print(f"🌸 {title.upper()}")
        print("="*70)
        print()

    def print_step(self, step_num: int, text: str):
        print(f"\n{'='*70}")
        print(f"📍 ШАГ {step_num}: {text}")
        print("="*70)

    def print_success(self, text: str):
        print(f"  ✅ {text}")

    def print_error(self, text: str):
        print(f"  ❌ {text}")

    def print_info(self, text: str):
        print(f"  ℹ️  {text}")

    def get_user_choice(self, prompt: str, options: list) -> int:
        print(f"\n{prompt}")
        print("-" * 50)

        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

        print(f"  0. Выход")

        while True:
            try:
                choice = int(input(f"\n👉 Ваш выбор (0-{len(options)}): "))
                if choice == 0:
                    print("\n👋 Программа завершена")
                    sys.exit(0)
                elif 1 <= choice <= len(options):
                    return choice
                else:
                    print(f"❌ Введите число от 0 до {len(options)}")
            except ValueError:
                print("❌ Введите корректное число")
            except KeyboardInterrupt:
                print("\n\n👋 Завершено")
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

        print(f"\n📂 Результаты: {self.output_dir}")
        return self.output_dir

    def auto_detect_datasets(self):
        """Автоматическое обнаружение датасетов"""
        print("\n🔍 Поиск датасетов...")

        available = []

        tabular_dir = Path("data/tabular")
        if tabular_dir.exists():
            for csv_file in tabular_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    available.append({
                        'type': 'tabular',
                        'name': csv_file.stem,
                        'path': csv_file,
                        'rows': len(df),
                        'cols': len(df.columns)
                    })
                    print(f"   ✅ {csv_file.name} ({len(df)} строк)")
                except:
                    pass

        return available

    def load_iris_builtin(self):
        """Загрузка встроенного Iris"""
        print("\n📥 Загрузка Iris Dataset...")

        from sklearn.datasets import load_iris
        iris = load_iris()

        self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.data['target'] = iris.target
        self.data['target_name'] = self.data['target'].map({
            0: 'setosa', 1: 'versicolor', 2: 'virginica'
        })

        self.target_column = 'target_name'
        self.print_success("Iris Dataset загружен")
        return True

    def load_csv_file(self, filepath: Path):
        print(f"\n📥 Загрузка {filepath.name}...")

        try:
            self.data = pd.read_csv(filepath)
            self.print_success(f"Загружено {len(self.data)} строк")

            # Автопоиск target
            target_candidates = ['target', 'class', 'label', 'category',
                               'target_name', 'flower_name', 'species']

            for col in target_candidates:
                if col in self.data.columns:
                    self.target_column = col
                    print(f"💡 Найдена target колонка: {col}")
                    break

            if self.target_column is None:
                print("\n📊 Колонки:")
                for i, col in enumerate(self.data.columns, 1):
                    print(f"   {i}. {col}")

                choice = int(input(f"\n👉 Выберите target (1-{len(self.data.columns)}): "))
                self.target_column = self.data.columns[choice - 1]

            return True

        except Exception as e:
            self.print_error(f"Ошибка: {e}")
            return False

    def load_data(self):
        self.print_step(1, "Загрузка данных")

        available = self.auto_detect_datasets()

        if available:
            print("\n" + "="*70)
            print("📂 НАЙДЕНЫ ДАТАСЕТЫ")
            print("="*70)

            options = []
            for i, ds in enumerate(available, 1):
                print(f"   {i}. {ds['name']} ({ds['rows']} строк)")
                options.append(ds)

            options.append({'type': 'builtin_iris', 'name': 'Iris Dataset (встроенный)'})
            print(f"   {len(options)}. Iris Dataset (встроенный)")
            options.append({'type': 'create_test', 'name': 'Создать тестовый датасет Iris (150 строк)'})
            print(f"   {len(options)}. Создать тестовый датасет Iris (150 строк)")

            choice = self.get_user_choice("\nВыберите датасет:",
                                         [opt['name'] for opt in options])

            selected = options[choice - 1]

            if selected['type'] == 'builtin_iris':
                return self.load_iris_builtin()
            elif selected['type'] == 'create_test':
                filepath = create_quick_test_dataset()
                return self.load_csv_file(filepath)
            else:
                return self.load_csv_file(selected['path'])

        else:
            print("\n⚠️  Датасеты не найдены в data/tabular/")

            choice = self.get_user_choice("\nЧто делать?", [
                "Создать тестовый датасет Iris (150 строк)",
                "Загрузить встроенный Iris Dataset",
                "Положить CSV в data/tabular/ и перезапустить"
            ])

            if choice == 1:
                filepath = create_quick_test_dataset()
                return self.load_csv_file(filepath)
            elif choice == 2:
                return self.load_iris_builtin()
            else:
                print("\n💡 Положите CSV файл в папку data/tabular/")
                print("   Затем перезапустите программу")
                return False

    def select_target_column(self):
        self.print_step(2, "Целевая переменная")

        if self.target_column and self.target_column in self.data.columns:
            print(f"\n✅ Выбрана: {self.target_column}")
        else:
            columns = list(self.data.columns)
            print("\n📊 Колонки:")

            for i, col in enumerate(columns, 1):
                sample = self.data[col].iloc[0] if len(self.data) > 0 else "N/A"
                unique_vals = self.data[col].nunique()
                print(f"  {i:2}. {col:30} | тип: {str(self.data[col].dtype):10} | уникальных: {unique_vals:4}")

            choice = self.get_user_choice("\nКакую предсказываем?", columns)
            self.target_column = columns[choice - 1]

        # 🔧 ИСПРАВЛЕНИЕ: Выбираем ТОЛЬКО numeric колонки для X
        self.y = self.data[self.target_column].values

        # Выбираем все колонки кроме target
        all_feature_cols = [col for col in self.data.columns if col != self.target_column]

        # Фильтруем только numeric
        self.feature_columns = []
        for col in all_feature_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.feature_columns.append(col)
            else:
                print(f"  ⚠️  Пропущена нечисловая колонка: {col} ({self.data[col].dtype})")

        if len(self.feature_columns) == 0:
            self.print_error("Нет числовых колонок для обучения!")
            return False

        self.X = self.data[self.feature_columns].values

        print(f"\n📊 Признаков (numeric): {len(self.feature_columns)}")
        print(f"   Колонки: {', '.join(self.feature_columns)}")
        print(f"   Образцов: {self.X.shape[0]}")
        print(f"   Классов: {len(np.unique(self.y))}")

        return True

    def analyze_data(self):
        """Анализ"""
        self.print_step(3, "Анализ данных")

        print("\n🔍 Анализ данных...")

        profiler = DataProfiler(dataset_name="User_Dataset")
        self.profile = profiler.profile_tabular_data(
            self.X,
            self.y,
            feature_names=self.feature_columns  # 🔧 Используем только numeric колонки
        )

        profiler.print_summary()

        profile_path = self.output_dir / "profile.json"
        self.profile.save(str(profile_path))

        try:
            viz_path = self.output_dir / "visualizations" / "data_profile.png"
            profiler.visualize_profile(save_path=str(viz_path))
            plt.close('all')
            self.print_success(f"График: {viz_path}")
        except Exception as e:
            print(f"  ⚠️  Ошибка визуализации: {e}")

        input("\n👉 Enter...")
        return True

    def select_and_train_model(self):
        """Выбор модели"""
        self.print_step(4, "Обучение моделей")

        self.selector = AdaptiveModelSelector()
        self.selector.create_default_models()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}")

        self.best_model = self.selector.profile_and_select(
            self.X, self.y,
            data_profile=self.profile.to_dict(),
            cv_folds=5
        )

        print(f"\n✅ Лучшая: {self.best_model.name}")
        input("\n👉 Enter...")
        return True

    def evaluate_and_show_results(self):
        """Результаты"""
        self.print_step(5, "Результаты")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        y_pred = self.selector.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "="*70)
        print("📈 ИТОГИ")
        print("="*70)
        print(f"\n🏆 Модель: {self.best_model.name}")
        print(f"📊 Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📋 Отчет:")
        print(classification_report(y_test, y_pred))

        try:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Matrix - {self.best_model.name}')
            cm_path = self.output_dir / "visualizations" / "confusion_matrix.png"
            plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
            plt.close('all')
            self.print_success(f"Matrix: {cm_path}")
        except Exception as e:
            print(f"  ⚠️  Ошибка: {e}")

        self.results = {
            'session_id': self.session_id,
            'target_column': self.target_column,
            'best_model': self.best_model.name,
            'accuracy': float(accuracy),
            'n_samples': int(self.X.shape[0]),
            'n_features': int(self.X.shape[1]),
            'timestamp': str(datetime.now())
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        self.print_success(f"Results: {results_path}")

        models_dir = self.output_dir / "models"
        self.selector.save_all_models(str(models_dir))
        self.print_success(f"Models: {models_dir}")

        input("\n👉 Enter...")
        return True

    def predict_new_data(self):
        self.print_step(6, "Предсказание")

        choice = self.get_user_choice("\nДействие:", [
            "Ввести данные",
            "Загрузить CSV",
            "Завершить"
        ])

        if choice == 1:
            print(f"\n📝 Введите {len(self.feature_columns)} значений ({', '.join(self.feature_columns)}):")
            try:
                values = [float(v) for v in input("👉 > ").strip().replace(',', ' ').split()]
                if len(values) != len(self.feature_columns):
                    print(f"❌ Ожидалось {len(self.feature_columns)} значений")
                    return False

                prediction = self.selector.predict(np.array([values]))[0]
                print(f"\n✅ Класс: {prediction}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")

        elif choice == 2:
            filepath = input("\n📂 CSV путь: ").strip()
            if Path(filepath).exists():
                try:
                    new_data = pd.read_csv(filepath)
                    # 🔧 Используем только numeric колонки
                    new_X = new_data[self.feature_columns].values
                    predictions = self.selector.predict(new_X)
                    new_data['prediction'] = predictions
                    output_path = self.output_dir / "predictions.csv"
                    new_data.to_csv(output_path, index=False)
                    self.print_success(f"Saved: {output_path}")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")

        return True

    def show_summary(self):
        self.print_header("Готово!")
        print("✅ Успешно!")
        print(f"\n📂 {self.output_dir}")
        print(f"🏆 {self.best_model.name}")
        print(f"📊 Accuracy: {self.results['accuracy']*100:.2f}%")
        print(f"\n💡 outputs/latest/")
        print("\n🎉 Спасибо!")

    def run(self):
        try:
            self.print_header("Adaptive ML System")
            print("Автоматический выбор ML моделей")
            print()
            print("📋 Как использовать:")
            print("   1. Положите CSV в data/tabular/")
            print("   2. ИЛИ создайте тестовый датасет")
            print("   3. ИЛИ используйте встроенный Iris")
            print("   DEVELOP TRICK: try scripts/download_all_datasets.py to get more data for tests")
            print()
            input("👉 Enter для начала...")

            self.setup_output_directory()

            if not self.load_data(): return False
            if not self.select_target_column(): return False
            if not self.analyze_data(): return False
            if not self.select_and_train_model(): return False
            if not self.evaluate_and_show_results(): return False
            if not self.predict_new_data(): return False

            self.show_summary()
            return True

        except KeyboardInterrupt:
            print("\n\n👋 Завершено")
            return False
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    app = AdaptiveMLApp()
    success = app.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()