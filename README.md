# dj-zarr-codecs

DataJoint codecs for storing arrays in [Zarr](https://zarr.dev/) format with schema-addressed paths.

## Overview

This package provides DataJoint codecs that store numpy arrays as Zarr format in object storage, using DataJoint's schema-addressed storage system. This creates browsable, organized storage that mirrors your database structure.

## Features

- **Schema-addressed storage**: Paths mirror database structure (`{schema}/{table}/{pk}/{field}.zarr`)
- **Zarr format**: Portable, cloud-optimized array storage with chunking and compression
- **Lazy loading**: Efficient access to large arrays without loading entire datasets
- **Direct access**: Use `zarr.open(ref.fsmap)` for advanced Zarr features
- **Automatic registration**: Codecs are automatically available after installation

## Installation

```bash
pip install dj-zarr-codecs
```

## Quick Start

```python
import datajoint as dj
import numpy as np

schema = dj.Schema('my_schema')

@schema
class Recording(dj.Manual):
    definition = """
    recording_id : int
    ---
    waveform : <zarr@>           # Stored as Zarr array
    """

# Insert numpy array
Recording.insert1({
    'recording_id': 1,
    'waveform': np.random.randn(1000, 32),
})

# Fetch returns Zarr array (read-only)
zarr_array = (Recording & {'recording_id': 1}).fetch1('waveform')

# Use directly with numpy
result = np.mean(zarr_array, axis=0)

# Or access as Zarr for advanced features
print(zarr_array.shape)   # (1000, 32)
print(zarr_array.chunks)  # Zarr chunking info
```

## Configuration

Configure your object storage in DataJoint:

```python
dj.config['stores'] = {
    'mystore': {
        'protocol': 's3',
        'endpoint': 's3.amazonaws.com',
        'bucket': 'my-bucket',
        'location': 'datajoint',
    }
}
```

## Codecs

### `<zarr@>` / `<zarr@store>`

Store numpy arrays in Zarr format with schema-addressed paths.

**Features:**
- Portable Zarr format (readable by any Zarr library)
- Efficient chunked storage
- Optional compression
- Schema-addressed paths for organization

**Usage:**
```python
class MyTable(dj.Manual):
    definition = """
    id : int
    ---
    data : <zarr@>          # Default store
    large_data : <zarr@s3>  # Specific store
    """
```

## Development

### Setup

```bash
git clone https://github.com/datajoint/dj-zarr-codecs.git
cd dj-zarr-codecs
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src tests
ruff format src tests
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License. Copyright (c) 2026 DataJoint Inc. See [LICENSE](LICENSE) for details.

## Related Projects

- [DataJoint](https://datajoint.com) - Framework for scientific data pipelines
- [Zarr](https://zarr.dev/) - Chunked, compressed, N-dimensional arrays
- [datajoint-python](https://github.com/datajoint/datajoint-python) - DataJoint for Python

## Support

- [GitHub Issues](https://github.com/datajoint/dj-zarr-codecs/issues)
- [DataJoint Slack](https://datajoint.com/slack)
