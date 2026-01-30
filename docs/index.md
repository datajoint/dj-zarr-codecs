# dj-zarr-codecs

DataJoint codecs for storing arrays in [Zarr](https://zarr.dev/) format with
schema-addressed paths.

## Overview

This package provides DataJoint codecs that store numpy arrays as Zarr format in
object storage, using DataJoint's schema-addressed storage system. This creates
browsable, organized storage that mirrors your database structure.

## Features

- **Schema-addressed storage**: Paths mirror database structure
  (`{schema}/{table}/{pk}/{field}.zarr`)
- **Zarr format**: Portable, cloud-optimized array storage with chunking and
  compression
- **Lazy loading**: Efficient access to large arrays without loading entire
  datasets
- **Automatic registration**: Codecs are automatically available after
  installation

## Installation

```bash
pip install dj-zarr-codecs
```

## Quick Start

```python
import datajoint as dj
import numpy as np

schema = dj.Schema("my_schema")


@schema
class Recording(dj.Manual):
    definition = """
    recording_id : int
    ---
    waveform : <zarr@store>           # Stored as Zarr array
    """


# Insert numpy array
Recording.insert1(
    {
        "recording_id": 1,
        "waveform": np.random.randn(1000, 32),
    }
)

# Fetch returns Zarr array (read-only)
zarr_array = (Recording & {"recording_id": 1}).fetch1("waveform")

# Use directly with numpy
result = np.mean(zarr_array, axis=0)

# Or access Zarr features
print(zarr_array.shape)   # (1000, 32)
print(zarr_array.chunks)  # Zarr chunking info
```

## Configuration

Configure your object storage in DataJoint:

```python
dj.config["stores"] = {
    "store": {
        "protocol": "s3",
        "endpoint": "s3.amazonaws.com",
        "bucket": "my-bucket",
        "location": "datajoint",
    }
}
```

## Usage

### `<zarr@store>`

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
    data : <zarr@store>          # Uses 'store' from dj.config["stores"]
    large_data : <zarr@archive>  # Uses 'archive' store
    """
```

## Storage Structure

Arrays are stored with schema-addressed paths:

```
{store_root}/{schema}/{table}/{pk}/{field}.zarr/
```

For example, a recording with `recording_id=1` in schema `neuro` would be stored at:

```
s3://my-bucket/datajoint/neuro/recording/recording_id=1/waveform.zarr/
```

This structure makes it easy to browse and manage stored data outside of DataJoint.
