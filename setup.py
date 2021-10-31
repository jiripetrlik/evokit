from setuptools import setup, find_packages
setup(
    name="evokit",
    version="0.1",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    author="Jiri Petrlik",
    author_email="jiripetrlik@gmail.com",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0"
    ],
)
