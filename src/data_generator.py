import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path("data")
        self.data_dir = data_dir
        self.tabular_dir = data_dir / "tabular"
        self.tabular_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Datagenerator initialized with directory: {self.tabular_dir}")

    def create_iris_dataset(self) -> Path:
        from sklearn.datasets import load_iris
        logger.info(f"Loading Iris dataset...")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        filepath = self.tabular_dir / "iris.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Iris dataset created: {filepath}")
        return filepath

    def create_wine_dataset(self) -> Path:
        from sklearn.datasets import load_wine
        logger.info("Creating Wine dataset")
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        df['target_name'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
        filepath = self.tabular_dir / "wine.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Wine dataset created: {filepath}")
        return filepath

    def create_digits_dataset(self) -> Path:
        from sklearn.datasets import load_digits
        logger.info("Creating Digits dataset")
        digits = load_digits()
        df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
        df['target'] = digits.target
        filepath = self.tabular_dir / "digits.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Digits dataset created: {filepath}")
        return filepath

    def create_titanic_dataset(self) -> Path:
        logger.info("Creating Titanic dataset")
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
        logger.info(f"Titanic dataset created: {filepath}")
        return filepath

    def create_synthetic_dataset(self) -> Path:
        from sklearn.datasets import make_blobs
        logger.info("Creating Synthetic dataset")
        np.random.seed(42)
        X, y = make_blobs(n_samples=500, n_features=5, centers=3, cluster_std=1.5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        df['target_name'] = df['target'].map({0: 'class_A', 1: 'class_B', 2: 'class_C'})
        filepath = self.tabular_dir / "synthetic.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Synthetic dataset created: {filepath}")
        return filepath

    def create_large_dataset(self) -> Path:
        from sklearn.datasets import make_classification
        logger.info("Creating Large dataset (100,000 rows)")
        X, y = make_classification(n_samples=100000, n_features=20, n_informative=15, n_redundant=5, n_classes=5,
                                   n_clusters_per_class=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        filepath = self.tabular_dir / "large_test.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Large dataset created: {filepath}")
        return filepath

    def auto_detect_datasets(self) -> List[Dict]:
        logger.debug(f"Auto-detecting datasets in {self.tabular_dir}")
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
                    logger.debug(f"Found dataset: {csv_file.stem} ({len(df)} rows)")
                except Exception as e:
                    logger.warning(f"Failed to read {csv_file}: {e}")
        logger.info(f"Found {len(available)} datasets")
        return available

    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading dataset: {filepath}")
        return pd.read_csv(filepath)

    def create_all_test_datasets(self) -> List[Path]:
        logger.info("Creating all test datasets")
        paths = []
        paths.append(self.create_iris_dataset())
        paths.append(self.create_wine_dataset())
        paths.append(self.create_digits_dataset())
        paths.append(self.create_titanic_dataset())
        paths.append(self.create_synthetic_dataset())
        logger.info(f"Created {len(paths)} test datasets")
        return paths