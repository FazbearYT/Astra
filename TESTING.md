# Тестирование на датасетах из открытых источников

Документ содержит пошаговую инструкцию по загрузке, подготовке и
запуску трёх реальных датасетов с UCI Machine Learning Repository.
Каждый датасет выбран так, чтобы продемонстрировать **разный профиль данных**
и показать, как **подсистема адаптации** реагирует на эти различия.

| Датасет | Размер | Особенности профиля | Что демонстрирует |
|---|---|---|---|
| Heart Disease (Cleveland) | 303 × 14 | Малый, бинарный, ~15% пропусков | Импьютация NaN, выбор на малых данных |
| Wine Quality (Red) | 1599 × 12 | Средний, 6 классов, экстремальный дисбаланс (0.015) | Многоклассовая классификация, дисбаланс |
| Adult Income (Census) | 32 561 × 15 | Большой, бинарный, 8 категориальных колонок | Препроцессинг категорий, адаптация SVM ядра |

---

## Перед запуском

```bash
python -m pip install -r requirements.txt
streamlit run web_app.py
```

После первого запуска `data/tabular/` будет автоматически заполнена
Iris/Titanic/Wine. Внешние датасеты добавляются в ту же папку.

---

## 1. Heart Disease (Cleveland)

**Источник:** [UCI ML Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Установка

PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path data\external | Out-Null
Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" `
  -OutFile "data\external\heart_raw.data" `
  -UseBasicParsing
```

Bash:
```bash
mkdir -p data/external
curl -o data/external/heart_raw.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
```

### Подготовка

Файл идёт без заголовков и с маркером пропусков `?`. Маленький скрипт:

```python
import pandas as pd
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv("data/external/heart_raw.data", header=None,
                 names=cols, na_values="?")
df["target"] = (df["target"].astype(float) > 0).astype(int)
df.to_csv("data/tabular/heart_disease.csv", index=False)
```

После — `heart_disease.csv` появится в списке датасетов в UI.

### В приложении

1. Шаг 1: выбрать `heart_disease`.
2. Шаг 2: target = `target`, preset = **Standard**.
3. Шаг 3: запустить обучение.

### Ожидаемые результаты

**Препроцессинг:**
- NaN импьютируется в колонках `ca`, `thal` (median).

**Detected profile:**
- complexity = `medium`
- classes = 2
- class_balance_ratio = 0.85 (умеренный дисбаланс)
- recommended families: NeuralNetwork, RandomForest, GradientBoosting
- preprocessing needs: feature_scaling, skewness_correction, outlier_treatment

**Профильная адаптация:**
- KNN: `n_neighbors = 13` (≈√(303/2))

**Лучшая модель:**

| Модель | final | accuracy | f1 |
|---|---|---|---|
| **RandomForest** | **0.914** | **0.885** | **0.885** |
| LogisticRegression | 0.902 | 0.869 | 0.869 |
| SVM | 0.889 | 0.853 | 0.853 |

Что интересно: RF и LR идут плотно, итоговый разрыв 0.012 — система
честно показывает, что выбор «не сильно лучше» альтернативы. На малых
датасетах это типичная картина.

---

## 2. Wine Quality (Red)

**Источник:** [UCI — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

### Установка

PowerShell:
```powershell
Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" `
  -OutFile "data\external\wine_quality_red.csv" `
  -UseBasicParsing
```

Bash:
```bash
curl -o data/external/wine_quality_red.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

### Подготовка

Файл уже с заголовками, но разделитель — точка с запятой:

```python
import pandas as pd
df = pd.read_csv("data/external/wine_quality_red.csv", sep=";")
df = df.rename(columns={"quality": "target"})
df.to_csv("data/tabular/wine_quality_red.csv", index=False)
```

Альтернативно: можно загрузить оригинальный файл через **upload-форму**
приложения — авто-определение разделителя сработает и `;` будет
распознан автоматически.

### В приложении

1. Шаг 1: выбрать `wine_quality_red`.
2. Шаг 2: target = `target`, preset = **Standard**.
3. Шаг 3: обучение.

### Ожидаемые результаты

**Detected profile:**
- complexity = `low`
- classes = **6** (значения качества 3-8)
- class_balance_ratio = **0.015** — экстремальный дисбаланс (классы 3 и 8 — единичные образцы)
- recommended families: SVM, RandomForest, **BalancedRandomForest**, LogisticRegression
- profile флагнул дисбаланс и предложил BalancedRandomForest

**Профильная адаптация:**
- KNN: `n_neighbors = 25` (≈√(1599/2))

**Лучшая модель:**

| Модель | final | accuracy | f1 |
|---|---|---|---|
| **KNN** | **0.751** | **0.672** | **0.657** |
| GaussianNB | 0.680 | 0.572 | 0.576 |
| RandomForest | 0.659 | 0.663 | 0.644 |

Что интересно: для wine quality multiclass с дисбалансом ~67% — это
**ожидаемая планка**, baseline из литературы. Никакая модель из
коробочного sklearn не вытащит больше без feature engineering.
**Это полезный «отрицательный» результат** — на защите можно честно
сказать: «система корректно определила сложность задачи и выбрала
оптимальную модель из доступных».

---

## 3. Adult Income (Census)

**Источник:** [UCI — Adult](https://archive.ics.uci.edu/dataset/2/adult)

### Установка

PowerShell:
```powershell
Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" `
  -OutFile "data\external\adult_raw.data" `
  -UseBasicParsing
```

Bash:
```bash
curl -o data/external/adult_raw.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

### Подготовка

Без заголовков, маркер пропусков ` ?` с ведущим пробелом:

```python
import pandas as pd
cols = ["age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","target"]
df = pd.read_csv("data/external/adult_raw.data", header=None,
                 names=cols, na_values=" ?", skipinitialspace=True)
df["target"] = (df["target"].str.strip() == ">50K").astype(int)
df.to_csv("data/tabular/adult_income.csv", index=False)
```

### В приложении

1. Шаг 1: выбрать `adult_income`.
2. Шаг 2: target = `target`, preset = **Standard**.
3. Шаг 3: обучение (займёт ~30-60 секунд из-за объёма).

### Ожидаемые результаты

**Препроцессинг (самое богатое из всех трёх):**
- 8 категориальных колонок:
  - one-hot (≤10 уникальных): `workclass`, `marital_status`,
    `relationship`, `race`, `sex`
  - label-encoded (>10 уникальных): `education`, `occupation`,
    `native_country`
- Итого: **15 колонок → 38 признаков**.

**Detected profile:**
- complexity = `low` (отношение n/p велико)
- classes = 2
- class_balance_ratio = 0.317 (заметный дисбаланс, ~24% позитивов)
- recommended families: SVM, RandomForest, BalancedRandomForest, LogisticRegression

**Профильная адаптация (две одновременно!):**
- SVM: `kernel rbf → linear` (n_samples > 20 000)
- KNN: `n_neighbors = 25`

**Лучшая модель — самый интересный кейс всех трёх:**

| Модель | final | metric | profile | accuracy | f1 |
|---|---|---|---|---|---|
| **GaussianNB** | **0.843** | 0.791 | **1.00** | 0.800 | 0.771 |
| RandomForest | 0.810 | ≈0.857 | 0.67 | **0.860** | **0.850** |
| DecisionTree | 0.810 | ≈0.855 | 0.67 | 0.858 | 0.855 |

**Это идеальный академический пример работы подсистемы адаптации.**

RandomForest даёт **более высокую точность (86% vs 80%)**, но
GaussianNB всё равно выбран лучшим — потому что:
- его профиль *идеально* совпадает с данными (complexity=low, требований мало, балл 1.00),
- а у RF профиль совпадает только частично (RF требует complexity=medium, но реальный профиль — low).

Формула `final = 0.75 × metric + 0.25 × profile`:
- GaussianNB: 0.75 × 0.791 + 0.25 × 1.00 = **0.843**
- RandomForest: 0.75 × 0.857 + 0.25 × 0.67 = **0.810**

**Это и есть тематический смысл проекта в одной таблице.** Без
профильной компоненты выбор шёл бы только по метрикам, и RandomForest
выиграл бы. Адаптационная подсистема явно меняет решение в пользу
модели, чьи теоретические предпосылки лучше соответствуют данным —
это ровно та история, которую заявляет тема работы.

---

## Сводный итог по трём датасетам

| Датасет | Профиль | Адаптации | Лучший по final | Лучший по accuracy |
|---|---|---|---|---|
| Heart Disease | medium, bal=0.85 | KNN n=13 | RandomForest (0.914) | RandomForest |
| Wine Quality | low, multiclass, bal=0.015 | KNN n=25 | KNN (0.751) | KNN |
| Adult Income | low, bal=0.32 | SVM→linear, KNN n=25 | **GaussianNB (0.843)** | RandomForest |

Три датасета показывают три разных характера работы системы:

1. **Heart Disease** — «обычный» случай: профиль и метрики согласны,
   выбор очевидный (RandomForest), близкая конкуренция.
2. **Wine Quality** — «честно сложный» случай: даже лучшая модель
   набирает ~67%, система корректно фиксирует ограничение задачи.
3. **Adult Income** — **«адаптация важнее метрик»**: профильная
   компонента сместила выбор в пользу теоретически более подходящей
   модели, хотя в чистых метриках лидировал RandomForest.

---

## Что показывать на защите

1. **Открыть скриншоты** трёх запусков (профильная панель + панель
   объяснения «Why this model was selected»).
2. **Указать на Adult Income** как наиболее яркую иллюстрацию темы:
   - "Без подсистемы адаптации был бы выбран RandomForest по accuracy".
   - "Подсистема выбрала GaussianNB — модель, чьи требования к профилю
     полностью выполняются".
   - "Это компромисс между интерпретируемостью адаптации и raw-точностью —
     осознанный design choice, отражённый в формуле скоринга".
3. **Открыть expander «Profile-driven hyperparameter adaptation»** —
   показать конкретные адаптации (KNN n_neighbors, SVM kernel switch).

---

## Расширение тестов

Для дополнительного покрытия рекомендуются:

- **Breast Cancer Wisconsin** (UCI) — 569 × 30, бинарная, должна дать
  ~97% accuracy, проверяет работу на высокоразмерных данных.
- **California Housing** (sklearn.datasets) — для регрессии, проверяет
  Ridge/Lasso ветку селектора.
- **Pima Indians Diabetes** — 768 × 8, бинарная, ~75% baseline, лёгкий
  и каноничный.

Все три устанавливаются аналогично — либо через `Invoke-WebRequest`
с UCI, либо через `sklearn.datasets.fetch_*` + сохранение в CSV.
