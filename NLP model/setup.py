from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp-continual-learning",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive collection of NLP models supporting continuous learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-continual-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyterlab>=3.4.0",
            "ipywidgets>=7.7.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "tensorflow-gpu>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-ewc=scripts.train_ewc_sentiment:main",
            "train-replay=scripts.train_replay_classifier:main",
            "train-progressive=scripts.train_progressive_lm:main",
            "train-gem=scripts.train_gem_ner:main",
            "train-online=scripts.train_online_intent:main",
        ],
    },
)