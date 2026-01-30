"""
Copyright (c) 2026 Davis Bennett. All rights reserved.

dj-zarr-codecs: Zarr integration for DataJoint
"""

from __future__ import annotations

from ._version import version as __version__
from .codecs import ZarrArrayCodec

__all__ = ["ZarrArrayCodec", "__version__"]
