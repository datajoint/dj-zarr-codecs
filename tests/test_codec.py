from __future__ import annotations

from typing import Any

import datajoint as dj
import numpy as np
import pytest
import zarr

from dj_zarr_codecs import ZarrArrayCodec

# The codec context, which DataJoint 2.3.4 and later pass

# How DataJoint 2.3 still fills `key`; the codec does not read the underscore keys.
LEGACY_KEY = {
    "id": 1,
    "_schema": "s",
    "_table": "t",
    "_field": "f",
    "_config": dj.config,
}


@pytest.mark.parametrize(
    "context",
    [None, {"schema": "s", "table": "t", "field": "f"}],
    ids=["no context", "no config"],
)
def test_encode_needs_context(context: dict[str, Any] | None) -> None:
    with pytest.raises(dj.DataJointError, match=r"DataJoint 2\.3\.4"):
        ZarrArrayCodec().encode(
            np.zeros((4, 4)), key=LEGACY_KEY, context=context, store_name="store"
        )


def test_decode_needs_context() -> None:
    stored = {"path": "x.zarr", "store": None, "shape": [4, 4], "dtype": "float64"}
    with pytest.raises(dj.DataJointError, match=r"missing: config"):
        ZarrArrayCodec().decode(stored, key={"_config": dj.config}, context=None)


# Through DataJoint, with a database and a file store


def test_numpy_array_roundtrip(schema: dj.Schema) -> None:
    """Test that we can round-trip a NumPy array through DataJoint using ZarrArrayCodec."""

    @schema
    class Table(dj.Manual):  # type: ignore[misc]
        definition = """
        id : int
        ---
        data : <zarr@store>
        """

    # Insert NumPy array
    rng = np.random.default_rng(42)
    test_array = rng.standard_normal((100, 32))
    Table.insert1(
        {
            "id": 1,
            "data": test_array,
        }
    )

    # Fetch returns the array
    fetched = (Table & {"id": 1}).fetch1("data")  # type: ignore[operator]

    # Verify roundtrip
    np.testing.assert_array_equal(fetched, test_array)


def test_zarray_roundtrip(schema: dj.Schema) -> None:
    """Test that we can round-trip a Zarr array through DataJoint."""

    @schema
    class Table(dj.Manual):  # type: ignore[misc]
        definition = """
        id : int
        ---
        data : <zarr@store>
        """

    # Insert Zarr array
    rng = np.random.default_rng(42)
    zarr_store: dict[str, bytes] = {}
    test_array = zarr.create_array(
        store=zarr_store, data=rng.standard_normal((100, 32))
    )

    Table.insert1(
        {
            "id": 1,
            "data": test_array,
        }
    )

    # Fetch returns the array
    fetched = (Table & {"id": 1}).fetch1("data")  # type: ignore[operator]

    # Verify roundtrip
    np.testing.assert_array_equal(fetched, test_array)
