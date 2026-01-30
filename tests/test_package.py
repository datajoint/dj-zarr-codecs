from __future__ import annotations

import importlib.metadata

import dj_zarr_codecs as m


def test_version():
    assert importlib.metadata.version("dj_zarr_codecs") == m.__version__
