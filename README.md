# Adaptive ML System - Подсистема адаптации моделей под рабочие профили

Автоматическая система выбора и применения ML-моделей на основе профилирования данных.

## 🎯 Описание

Система автоматически:
1. **Анализирует** входные данные (профилирование)
2. **Выбирает** оптимальную модель из набора предобученных
3. **Обучает** и **тестирует** выбранную модель
4. **Предсказывает** результаты с минимальным участием пользователя

## 📁 Структура проекта
adaptive_ml_system/ 

need upd later :)

## 🚀 Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone <repository_url>
cd <repository_name>

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## 🚀 Быстрый Запуск

```bash
# 1. Установка
pip install -r requirements.txt

# 2. Обучение на Iris (20-60 сек)
python scripts/train_iris_models.py

# 3. Загрузка Oxford Flowers (1-3 минуты)
python scripts/download_oxford_flowers.py

# 4. Обучение YOLO (ДОЛГО! Может занять несколько часов)
python scripts/train_yolo_flowers.py

# 5. Оценка моделей
python scripts/evaluate_models.py

# 6. Тесты
pytest tests/ -v
```