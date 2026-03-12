"""
Setup script for Adaptive ML System
"""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="adaptive-ml-system",
    version="1.0.0",
    author="Adaptive ML Team",
    author_email="team@adaptiveml.system",
    description="Подсистема адаптации ML моделей под рабочие профили данных",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-ml-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "yolo": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "ultralytics>=8.0.0",
            "opencv-python>=4.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-ml-train=scripts.train_iris_models:main",
            "adaptive-ml-yolo=scripts.train_yolo_flowers:train_yolo_flowers",
            "adaptive-ml-evaluate=scripts.evaluate_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords="machine-learning adaptive-models auto-ml classification yolo",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/adaptive-ml-system/issues",
        "Source": "https://github.com/yourusername/adaptive-ml-system",
        "Documentation": "https://adaptive-ml-system.readthedocs.io/",
    },
)