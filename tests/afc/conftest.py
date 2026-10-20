"""GPU tests need CUDA and LLAVA_MODEL_PATH (a LLaVA-1.5-7B directory or hub
id resolvable offline); otherwise they are skipped, never faked."""

import os

import pytest
import torch

MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA + LLaVA-1.5-7B")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available() and MODEL_PATH:
        return
    skip = pytest.mark.skip(reason="needs CUDA and LLAVA_MODEL_PATH")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
