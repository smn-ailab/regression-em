"""setup.py for installing regressionEM package."""
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    with open("requirements.txt", 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name="regression_em",
    version="0.0.1",
    author="SMN a.i lab.",
    author_email="suguru_yaginuma@so-netmedia.jp",
    description="A small example package",
    url="https://github.com/smn-ailab/regression-em/",
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)
