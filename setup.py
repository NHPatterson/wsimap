#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from _setup_utils import get_requirements_and_links

# with open("/docs/readme.rst") as readme_file:
#     readme = readme_file.read()
#
# with open("/docs/history.rst") as history_file:
#     history = history_file.read()

requirements, dep_links = get_requirements_and_links("requirements.txt")

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Nathan Heath Patterson:",
    author_email="heath.patterson@vanderbilt.edu",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python package for preparing whole slide images from microscopy for deep learning segmentation",
    entry_points={
        "console_scripts": [
            "wsimap=wsimap.cli:main",
        ],
    },
    install_requires=requirements,
    dependency_links=[
        "http://github.com/facebookresearch/detectron2/tarball/master#egg=detectron2-0.1.1"
    ],
    license="Apache Software License 2.0",
    long_description="",
    include_package_data=True,
    keywords="wsimap",
    name="wsimap",
    packages=find_packages(include=["wsimap", "wsimap.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/nhpatterson/wsimap",
    version="0.0.1",
    zip_safe=False,
)
