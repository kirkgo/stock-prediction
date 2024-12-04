from setuptools import setup, find_packages

setup(
    name="stock_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.3",
        "matplotlib>=3.9.3",
        "seaborn>=0.13.0",
        "tensorflow>=2.18.0",
        "yfinance>=0.2.49",
        "scikit-learn>=1.5.2",
        "fastapi>=0.115.5",
        "prometheus_client>=0.21.1",
        "pydantic>=2.10.3",
        "PyYAML>=6.0.2",
    ],
    python_requires=">=3.8",
)
