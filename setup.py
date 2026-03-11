from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="luganda-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="you@example.com",
    description="End-to-end pipeline for building a Luganda–English paired speech dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/luganda-pipeline",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "accelerate>=0.30.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "sentencepiece>=0.2.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.1",
        "click>=8.1.7",
        "sacrebleu>=2.4.0",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
        "tts-coqui": ["TTS>=0.22.0"],
        "vad": ["silero-vad>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "luganda-pipeline=luganda_pipeline.pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    include_package_data=True,
    package_data={
        "luganda_pipeline": ["py.typed"],
    },
)
