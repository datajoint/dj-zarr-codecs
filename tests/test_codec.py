import datajoint as dj
import numpy as np
import zarr


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
