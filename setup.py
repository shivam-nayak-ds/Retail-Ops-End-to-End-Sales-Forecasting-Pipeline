from setuptools import setup, find_packages

def get_requirements(filepath: str) -> list[str]:
    """Read requirements.txt and return a list of packages."""
    requirements = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-e"):
                requirements.append(line)
    return requirements


setup(
    name="retail_ops_pipeline",
    version="1.0.0",
    author="Shivam Nayak",
    description="End-to-End Retail Sales Forecasting MLOps Pipeline targeting 34-40 LPA AI Engineer roles",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "retail-train=Retail_Ops_Pipeline.pipeline.training_pipeline:main",
            "retail-predict=Retail_Ops_Pipeline.pipeline.prediction_pipeline:main",
        ]
    },
)
