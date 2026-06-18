"""Shim for legacy ``pip install -e .`` / ``--no-index`` installs on Compute Canada.

All real metadata lives in ``pyproject.toml``.
"""

from setuptools import setup

setup()
