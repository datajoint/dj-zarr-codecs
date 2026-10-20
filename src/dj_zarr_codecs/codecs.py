"""Zarr codec implementation for DataJoint."""

from __future__ import annotations

from typing import Any

import datajoint as dj
import numpy as np
import zarr
from datajoint import DataJointError
from datajoint.builtin_codecs import SchemaCodec


def _require_context(context: dict[str, Any] | None, *names: str) -> dict[str, Any]:
    """
    Return ``context``, checked to carry ``names``.

    DataJoint 2.3.4 and later pass ``context`` to every codec that declares it:
    ``schema``, ``table``, ``field``, and ``config`` on insert, ``config`` on
    fetch. The codec reads them there only, never from underscore keys in ``key``.
    """
    missing = [n for n in names if context is None or context.get(n) is None]
    if context is None or missing:
        msg = (
            "<zarr> needs the context that DataJoint 2.3.4 and later pass to codecs "
            f"(missing: {', '.join(missing)}); install datajoint>=2.3.4"
        )
        raise DataJointError(msg)
    return context


def _primary_key(key: dict[str, Any] | None) -> dict[str, Any]:
    """The primary key values in ``key``, without the underscore keys DataJoint 2.3 adds."""
    return {k: v for k, v in (key or {}).items() if not k.startswith("_")}


class ZarrArrayCodec(SchemaCodec):  # type: ignore[misc]
    """
    Store NumPy arrays in Zarr format with schema-addressed paths.

    The ``<zarr@>`` codec stores NumPy arrays as Zarr format in object storage
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

        # Insert NumPy array
        Recording.insert1({
            'recording_id': 1,
            'waveform': np.random.randn(1000, 32),
        })

        # Fetch returns Zarr array (read-only)
        zarr_array = (Recording & {'recording_id': 1}).fetch1('waveform')

        # Use with NumPy
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
        Validate that value is a NumPy array suitable for Zarr storage.

        Parameters
        ----------
        value : Any
            Value to validate.

        Raises
        ------
        TypeError
            If value is not a NumPy array or Zarr array.
        DataJointError
            If array has object dtype.
        """
        if not isinstance(value, np.ndarray | zarr.Array):
            msg = f"<zarr> requires a NumPy array or Zarr array, got {type(value).__name__}"
            raise TypeError(msg)
        if value.dtype == object:
            msg = "<zarr> does not support object dtype arrays"
            raise DataJointError(msg)

    def encode(
        self,
        value: np.ndarray[Any, Any] | zarr.Array[Any],
        *,
        key: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Encode NumPy array as Zarr format in object storage.

        Parameters
        ----------
        value : np.ndarray
            NumPy array to store.
        key : dict, optional
            Primary key values for path construction.
        context : dict, optional
            ``schema``, ``table``, ``field``, and ``config`` of the insert, as
            DataJoint passes them.
        store_name : str, optional
            Name of the object store to use.

        Returns
        -------
        dict
            Metadata stored in database: path, store, codec_version, shape, dtype.

        Raises
        ------
        DataJointError
            If ``context`` lacks the ``schema``, ``table``, ``field``, or
            ``config`` that DataJoint 2.3.4 and later pass, or if encoding fails.
        """
        from dj_zarr_codecs import __version__  # noqa: PLC0415

        context = _require_context(context, "schema", "table", "field", "config")
        config = context["config"]

        # Build schema-addressed path
        path, _token = self._build_path(
            context["schema"],
            context["table"],
            context["field"],
            _primary_key(key),
            ext=".zarr",
            store_name=store_name,
            config=config,
        )

        # Get storage backend
        backend = self._get_backend(store_name, config=config)

        # Get fsspec mapper for direct Zarr write
        store_map = backend.get_fsmap(path)

        zarr.create_array(store=store_map, data=value, write_data=True)  # type: ignore[arg-type]

        # Return metadata for database storage (stored as JSON column)
        return {
            "path": path,
            "store": store_name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "provenance": {
                "datajoint-python": dj.__version__,
                "dj-zarr-codecs": __version__,
            },
        }

    def decode(
        self,
        stored: dict[str, Any],
        *,
        key: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> zarr.Array[Any]:
        """
        Decode Zarr array from object storage.

        Parameters
        ----------
        stored : dict
            Metadata from database containing path and store.
        key : dict, optional
            Primary key values (unused).
        context : dict, optional
            ``config`` of the fetching connection, as DataJoint passes it.

        Returns
        -------
        zarr.Array
            Read-only Zarr array. Use with NumPy operations or access
            Zarr-specific features.

        Raises
        ------
        DataJointError
            If ``context`` lacks the ``config`` that DataJoint 2.3.4 and later
            pass, or if decoding fails.
        """
        del key  # the stored metadata locates the array
        config = _require_context(context, "config")["config"]
        # Get storage backend
        backend = self._get_backend(stored.get("store"), config=config)

        # Get fsspec mapper for Zarr path
        store_map = backend.get_fsmap(stored["path"])

        # Open Zarr array (read-only)
        return zarr.open_array(store_map, mode="r")
