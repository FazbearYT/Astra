"""
🌸 Adaptive ML System - Единый интерфейс пользователя
=====================================================

Простое взаимодействие:
1. Запустите: python app.py
2. Следуйте инструкциям
3. Получите результат
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_profiler import DataProfiler
from model_selector import AdaptiveModelSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AdaptiveMLApp:
    """
    Главный класс приложения
    Простой и понятный интерфейс для пользователя
    """

    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.target_column = None
        self.profile = None
        self.selector = None
        self.best_model = None
        self.results = {}

    def clear_screen(self):
        """Очистка экрана"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        """Красивый заголовок"""
        self.clear_screen()
        print("=" * 70)
        print(f"🌸 {title.upper()}")
        print("=" * 70)
        print()

    def print_step(self, step_num: int, text: str):
        """Вывод шага"""
        print(f"\n{'=' * 70}")
        print(f"📍 ШАГ {step_num}: {text}")
        print("=" * 70)

    def get_user_choice(self, prompt: str, options: list) -> int:
        """Получение выбора пользователя"""
        print(f"\n{prompt}")
        print("-" * 50)

        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")

        while True:
            try:
                choice = int(input(f"\nВаш выбор (1-{len(options)}): "))
                if 1 <= choice <= len(options):
                    return choice
                else:
                    print(f"❌ Введите число от 1 до {len(options)}")
            except ValueError:
                print("❌ Введите корректное число")
            except KeyboardInterrupt:
                print("\n\n👋 Программа завершена пользователем")
                sys.exit(0)

    def load_csv_data(self):
        """Загрузка CSV файла"""
        self.print_step(1, "Загрузка данных")

        print("\n📂 Укажите путь к вашему CSV файлу:")
        print("   Пример: data/my_dataset.csv")
        print("   Или оставьте пустым для использования примера (Iris)")

        filepath = input("\nПуть к файлу: ").strip()

        if not filepath:
            print("\n📥 Загрузка примера (Iris Dataset)...")
            from sklearn.datasets import load_iris
            iris = load_iris()
            self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.data['target'] = iris.target
            self.target_column = 'target'
            print("✅ Пример загружен")
        else:
            if not Path(filepath).exists():
                print(f"❌ Файл не найден: {filepath}")
                return False

            try:
                print(f"\n📥 Загрузка {filepath}...")
                self.data = pd.read_csv(filepath)
                print(f"✅ Загружено {len(self.data)} строк, {len(self.data.columns)} колонок")
            except Exception as e:
                print(f"❌ Ошибка загрузки: {e}")
                return False

        # Показать первые строки
        print("\n📋 Первые 5 строк датасета:")
        print(self.data.head())

        return True

    def select_target_column(self):
        """Выбор целевой колонки"""
        self.print_step(2, "Выбор целевой переменной")

        print("\n📊 Доступные колонки:")
        columns = list(self.data.columns)

        for i, col in enumerate(columns, 1):
            # Показываем тип и пример значения
            sample = self.data[col].iloc[0] if len(self.data) > 0 else "N/A"
            print(f"  {i}. {col:30} (тип: {self.data[col].dtype}, пример: {sample})")

        choice = self.get_user_choice(
            "\nКакую колонку предсказываем? (это будет наш target)",
            columns
        )

        self.target_column = columns[choice - 1]
        print(f"\n✅ Выбрана целевая колонка: {self.target_column}")

        # Разделение на X и y
        self.y = self.data[self.target_column].values
        self.X = self.data.drop(columns=[self.target_column]).values

        print(f"   Признаков: {self.X.shape[1]}")
        print(f"   Образцов: {self.X.shape[0]}")

        return True

    def analyze_data(self):
        """Анализ данных"""
        self.print_step(3, "Анализ данных (профилирование)")

        print("\n🔍 Анализирую ваш датасет...")
        print("   • Статистики признаков")
        print("   • Распределение классов")
        print("   • Сложность данных")
        print("   • Корреляции")

        profiler = DataProfiler(dataset_name="User_Dataset")
        self.profile = profiler.profile_tabular_data(
            self.X,
            self.y,
            feature_names=[col for col in self.data.columns if col != self.target_column]
        )

        # Показать профиль
        profiler.print_summary()

        # Сохранить профиль
        profile_path = Path("models/user_profile.json")
        profile_path.parent.mkdir(exist_ok=True)
        self.profile.save(str(profile_path))

        input("\nНажмите Enter для продолжения...")
        return True

    def select_and_train_model(self):
        """Выбор и обучение модели"""
        self.print_step(4, "Выбор и обучение лучшей модели")

        print("\n🤖 Доступные модели:")
        print("   • Random Forest - для небольших сбалансированных данных")
        print("   • SVM - для данных с четкими границами")
        print("   • Gradient Boosting - для сложных паттернов")
        print("   • Neural Network - универсальная модель")
        print("   • Logistic Regression - быстрая базовая модель")

        print("\n⚙️  Автоматический выбор лучшей модели...")
        print("   (система протестирует несколько моделей и выберет лучшую)")

        self.selector = AdaptiveModelSelector()
        self.selector.create_default_models()

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n📊 Данные:")
        print(f"   Train: {len(X_train)} образцов")
        print(f"   Test: {len(X_test)} образцов")

        # Выбор модели
        self.best_model = self.selector.profile_and_select(
            self.X, self.y,
            data_profile=self.profile.to_dict(),
            cv_folds=5
        )

        print(f"\n✅ Лучшая модель выбрана: {self.best_model.name}")
        print(f"   Описание: {self.best_model.description}")

        input("\nНажмите Enter для продолжения...")
        return True

    def evaluate_and_show_results(self):
        """Оценка и показ результатов"""
        self.print_step(5, "Результаты")

        # Финальная оценка
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        y_pred = self.selector.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 70)

        print(f"\n🏆 Лучшая модель: {self.best_model.name}")
        print(f"📊 Точность на тесте: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        print(f"\n📋 Отчет о классификации:")
        print(classification_report(y_test, y_pred))

        # Сохранение результатов
        results = {
            'dataset': 'User_Dataset',
            'target_column': self.target_column,
            'best_model': self.best_model.name,
            'accuracy': float(accuracy),
            'n_samples': int(self.X.shape[0]),
            'n_features': int(self.X.shape[1]),
            'timestamp': str(datetime.now())
        }

        results_path = Path("models/user_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Результаты сохранены: {results_path}")

        # Сохранение моделей
        models_dir = Path("models/user_models")
        models_dir.mkdir(exist_ok=True)
        self.selector.save_all_models(str(models_dir))

        print(f"💾 Модели сохранены: {models_dir}")

        input("\nНажмите Enter для продолжения...")
        return True

    def predict_new_data(self):
        """Предсказание новых данных"""
        self.print_step(6, "Предсказание новых данных")

        print("\n🔮 Хотите сделать предсказание для новых данных?")

        choice = self.get_user_choice(
            "\nВыберите действие:",
            [
                "Ввести данные вручную",
                "Загрузить из CSV файла",
                "Пропустить и завершить"
            ]
        )

        if choice == 1:
            # Ввод вручную
            print(f"\n📝 Введите значения признаков ({self.X.shape[1]} значений):")
            print(f"   (через запятую или пробел)")

            try:
                values = input("   > ").strip()
                values = [float(v) for v in values.replace(',', ' ').split()]

                if len(values) != self.X.shape[1]:
                    print(f"❌ Ожидалось {self.X.shape[1]} значений, получено {len(values)}")
                    return False

                prediction = self.selector.predict(np.array([values]))
                proba = self.selector.predict_proba(np.array([values]))

                print(f"\n✅ Предсказание:")
                print(f"   Класс: {prediction[0]}")
                print(f"   Вероятности: {proba[0]}")

            except Exception as e:
                print(f"❌ Ошибка: {e}")
                return False

        elif choice == 2:
            # Загрузка из CSV
            filepath = input("\n📂 Путь к CSV файлу с данными: ").strip()

            if Path(filepath).exists():
                try:
                    new_data = pd.read_csv(filepath)
                    print(f"✅ Загружено {len(new_data)} образцов")

                    predictions = self.selector.predict(new_data.values)

                    print(f"\n✅ Предсказания:")
                    for i, pred in enumerate(predictions[:10]):  # Первые 10
                        print(f"   Образец {i + 1}: Класс {pred}")

                    if len(predictions) > 10:
                        print(f"   ... и еще {len(predictions) - 10}")

                    # Сохранение
                    new_data['prediction'] = predictions
                    output_path = Path("predictions.csv")
                    new_data.to_csv(output_path, index=False)
                    print(f"\n💾 Предсказания сохранены: {output_path}")

                except Exception as e:
                    print(f"❌ Ошибка: {e}")
                    return False
            else:
                print(f"❌ Файл не найден: {filepath}")
                return False

        return True

    def run(self):
        """Запуск приложения"""
        try:
            self.print_header("Adaptive ML System")
            print("Добро пожаловать в систему адаптивного выбора ML моделей!")
            print()
            print("Эта программа поможет вам:")
            print("  1. Загрузить ваш датасет")
            print("  2. Автоматически проанализировать данные")
            print("  3. Выбрать лучшую модель машинного обучения")
            print("  4. Получить предсказания")
            print()
            input("Нажмите Enter для начала...")

            # Шаг 1: Загрузка данных
            if not self.load_csv_data():
                return False

            # Шаг 2: Выбор target
            if not self.select_target_column():
                return False

            # Шаг 3: Анализ
            if not self.analyze_data():
                return False

            # Шаг 4: Выбор модели
            if not self.select_and_train_model():
                return False

            # Шаг 5: Результаты
            if not self.evaluate_and_show_results():
                return False

            # Шаг 6: Предсказания
            if not self.predict_new_data():
                return False

            # Завершение
            self.print_header("Готово!")
            print("✅ Все выполнено успешно!")
            print()
            print("📂 Созданные файлы:")
            print("  • models/user_profile.json - профиль данных")
            print("  • models/user_results.json - результаты")
            print("  • models/user_models/ - обученные модели")
            print()
            print("🎉 Спасибо за использование Adaptive ML System!")
            print()

            return True

        except KeyboardInterrupt:
            print("\n\n👋 Программа завершена пользователем")
            return False
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Точка входа"""
    app = AdaptiveMLApp()
    success = app.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()