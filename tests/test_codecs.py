import datajoint as dj
import numpy as np
import pytest
import zarr

from dj_zarr_codecs.codecs import ZarrCodec

@pytest.fixture(scope="function")
def schema(dj_connection: dj.Connection) -> dj.Schema:
    """
    Fixture to create and drop a test schema for each test function.
    """
    test_schema = dj.Schema('test_schema', connection=dj_connection)
    yield test_schema
    test_schema.drop(prompt=False)

def test_numpy_array_roundtrip(schema: dj.Schema) -> None:
    """
    Test that we can round-trip a numpy array through DataJoint using ZarrCodec
    """
    @schema
    class Table(dj.Manual):
        definition = """
        id : int
        ---
        data : <zarr@store>
        """

    # Insert numpy array
    test_array = np.random.randn(100, 32)
    Table.insert1({
        'id': 1,
        'data': test_array,
    })

    # Fetch returns the array
    fetched = (Table & {'id': 1}).fetch1('data')

    # Verify roundtrip
    np.testing.assert_array_equal(fetched, test_array)

def test_zarray_roundtrip(schema: dj.Schema) -> None:
    """
    Test that we can round-trip a Zarr array through DataJoint
    """
    @schema
    class Table(dj.Manual):
        definition = """
        id : int
        ---
        data : <zarr@store>
        """

    # Insert numpy array
    zarr_store = {}
    test_array = zarr.create_array(store=zarr_store, data=
        np.random.randn(100, 32)
        )

    Table.insert1({
        'id': 1,
        'data': test_array,
    })

    # Fetch returns the array
    fetched = (Table & {'id': 1}).fetch1('data')

    # Verify roundtrip
    np.testing.assert_array_equal(fetched, test_array)