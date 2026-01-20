import datajoint as dj
import numpy as np
import pytest


def test_array_roundtrip(dj_connection: dj.Connection) -> None:
    """
    Test that we can round-trip a Zarr array through DataJoint
    """
    schema = dj.Schema('test_schema', connection=dj_connection)

    @schema
    class Recording(dj.Manual):
        definition = """
        recording_id : int
        ---
        waveform : longblob           # Stored as blob for now
        """

    # Insert numpy array
    test_array = np.random.randn(100, 32)
    Recording.insert1({
        'recording_id': 1,
        'waveform': test_array,
    })

    # Fetch returns the array
    fetched = (Recording & {'recording_id': 1}).fetch1('waveform')

    # Verify roundtrip
    np.testing.assert_array_equal(fetched, test_array)

    # Cleanup
    schema.drop(force=True)


@pytest.mark.skip(reason="Zarr group support not yet implemented")
def test_group_roundtrip(dj_connection: dj.Connection) -> None:
    """
    Test that we can round-trip a Zarr group through DataJoint
    """
    pass
