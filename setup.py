from setuptools import setup, find_packages

setup(
    name="auto_circuit_martian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformer_lens",
        "nnsight",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "typing_extensions",
        "einops",
        "jaxtyping",
        "fancy_einsum",
        "plotly",
        "datasets",
        "wandb",
        "pytest",
        "black",
        "isort",
        "flake8",
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'isort',
            'flake8',
            'mypy',
        ],
    },
    author="Abir Harrasse",
    author_email="abirharrasse@gmail.com",
    description="Auto-circuit package with nnsight support for analyzing neural networks",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "circuits",
        "interpretability",
        "nnsight",
        "neural networks",
        "analysis"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    url="https://github.com/abirharrasse/auto-circuit_martian",
    project_urls={
        "Bug Reports": "https://github.com/abirharrasse/auto-circuit_martian/issues",
        "Source": "https://github.com/abirharrasse/auto-circuit_martian",
    },
)
