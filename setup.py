from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="electra_spacing",
    version="0.1",
    description="KoELECTRA based space correction model",
    author="seujung",
    author_email="developers@scatterlab.co.kr",
    install_requires=required,
    packages=find_packages(exclude=["docs", "tests", "tmp", "data"]),
    python_requires=">=3.6",
    license="Apache License 2.0",
    zip_safe=False,
    include_package_data=True,
)