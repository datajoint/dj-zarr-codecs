"""Zarr codec implementation for DataJoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr

try:
    import datajoint as dj
    from datajoint import DataJointError
    from datajoint.builtin_codecs import SchemaCodec
except ImportError as e:
    raise ImportError(
        "datajoint>=2.0.0a22 is required. Install with: pip install 'datajoint>=2.0.0a22'"
    ) from e


class ZarrCodec(SchemaCodec):
    """
    Store numpy arrays in Zarr format with schema-addressed paths.

    The ``<zarr@>`` codec stores numpy arrays as Zarr format in object storage
    using schema-addressed paths: ``{schema}/{table}/{pk}/{field}.zarr``.

    Features:
        - **Portable**: Standard Zarr format readable by any Zarr library
        - **Efficient**: Chunked storage with optional compression
        - **Organized**: Schema-addressed paths mirror database structure
        - **Direct access**: Returns Zarr arrays for use with zarr library

    Example::

        import datajoint as dj
        import numpy as np

        schema = dj.Schema('my_schema')

        @schema
        class Recording(dj.Manual):
            definition = '''
            recording_id : int
            ---
            waveform : <zarr@>           # default store
            spectrogram : <zarr@archive>  # specific store
            '''

        # Insert numpy array
        Recording.insert1({
            'recording_id': 1,
            'waveform': np.random.randn(1000, 32),
        })

        # Fetch returns Zarr array (read-only)
        zarr_array = (Recording & {'recording_id': 1}).fetch1('waveform')

        # Use with numpy
        result = np.mean(zarr_array, axis=0)

        # Access Zarr features
        print(zarr_array.shape)   # (1000, 32)
        print(zarr_array.chunks)  # Zarr chunk info
        print(zarr_array.dtype)   # Data type

    Storage Structure::

        {store_root}/{schema}/{table}/{pk}/{field}.zarr/

    Deletion:
        Requires garbage collection via ``dj.gc.collect()`` to remove
        orphaned Zarr arrays from storage.

    See Also
    --------
    SchemaCodec : Base class for schema-addressed codecs.
    datajoint.gc : Garbage collection for orphaned storage.
    """

    name = "zarr"

    def validate(self, value: Any) -> None:
        """
        Validate that value is a numpy array suitable for Zarr storage.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        DataJointError
            If value is not a numpy array or has object dtype.
        """
        if not isinstance(value, np.ndarray):
            raise DataJointError(
                f"<zarr> requires numpy.ndarray, got {type(value).__name__}"
            )
        if value.dtype == object:
            raise DataJointError("<zarr> does not support object dtype arrays")

    def encode(
        self,
        value: np.ndarray,
        *,
        key: dict | None = None,
        store_name: str | None = None,
    ) -> dict:
        """
        Encode numpy array as Zarr format in object storage.

        Parameters
        ----------
        value : np.ndarray
            Numpy array to store.
        key : dict, optional
            Primary key values for path construction.
        store_name : str, optional
            Name of the object store to use.

        Returns
        -------
        dict
            Metadata stored in database: path, store, shape, dtype.

        Raises
        ------
        DataJointError
            If encoding fails.
        """
        try:
            # Extract context from key
            schema, table, field, primary_key = self._extract_context(key)

            # Build schema-addressed path
            path, _token = self._build_path(
                schema, table, field, primary_key, ext=".zarr", store_name=store_name
            )

            # Get storage backend
            backend = self._get_backend(store_name)

            # Get fsspec mapper for direct Zarr write
            store_map = backend.get_fsmap(path)

            # Write array to Zarr format
            zarr.save_array(store_map, value)

            # Return metadata for database storage
            return {
                "path": path,
                "store": store_name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        except Exception as e:
            raise DataJointError(f"Failed to encode Zarr array: {e}") from e

    def decode(self, stored: dict, *, key: dict | None = None) -> zarr.Array:
        """
        Decode Zarr array from object storage.

        Parameters
        ----------
        stored : dict
            Metadata from database containing path and store.
        key : dict, optional
            Primary key values (unused).

        Returns
        -------
        zarr.Array
            Read-only Zarr array. Use with numpy operations or access
            Zarr-specific features.

        Raises
        ------
        DataJointError
            If decoding fails.
        """
        try:
            # Get storage backend
            backend = self._get_backend(stored.get("store"))

            # Get fsspec mapper for Zarr path
            store_map = backend.get_fsmap(stored["path"])

            # Open Zarr array (read-only)
            return zarr.open(store_map, mode="r")

        except Exception as e:
            raise DataJointError(f"Failed to decode Zarr array: {e}") from e
