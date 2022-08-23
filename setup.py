from setuptools import setup, find_packages

test_deps = [
    "pytest>=7.1.2",
]
extras = {
    'test': test_deps,
}

setup(
    name="evokit",
    version="0.1",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    author="Jiri Petrlik",
    author_email="jiripetrlik@gmail.com",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "numba>=0.56.0"
    ],
    tests_require=test_deps,
    extras_require=extras,
)
