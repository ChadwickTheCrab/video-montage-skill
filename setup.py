from setuptools import setup, find_packages

setup(
    name="video-montage-skill",
    version="0.1.0",
    description="AI-powered rough-cut video editor for professional montages",
    author="Pinch & Chad",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        # Phase 1: No external dependencies (stdlib + FFmpeg)
    ],
    extras_require={
        "enhanced": [
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "librosa>=0.10.0",
        ],
        "cloud": [
            "dropbox>=11.0.0",
            "requests>=2.31.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-montage=video_montage_skill.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Non-Linear Editor",
    ],
)
