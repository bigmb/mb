#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup,find_namespace_packages
from mb.version import version

setup(
    name="mb_base",
    version=version,
    description="Meta Package for mb_* packages",
    author=["Malav Bateriwala"],
    packages=find_namespace_packages(include=["mb.*"]),
    scripts=[],
    install_requires=[
        "mb_pandas",
        "mb_utils",
        "mb_sql"],
    python_requires='>=3.8',)
