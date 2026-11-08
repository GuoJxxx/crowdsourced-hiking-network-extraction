"""Locate GDAL resources in the running Python installation."""
from importlib.util import find_spec
import os
from pathlib import Path
import sys


def _package_data(name):
    spec = find_spec(name)
    if spec is not None and spec.origin:
        return Path(spec.origin).parent / 'gdal_data'
    return None


def configure_gdal_data():
    """Set the resource directory before Rasterio or vector drivers are imported."""
    candidates = [_package_data('rasterio')]
    # Conda places these files under Library/share/gdal on Windows.
    # Wheels keep them beside their Python modules.
    for prefix in dict.fromkeys((sys.prefix, sys.base_prefix)):
        candidates.extend((Path(prefix) / 'Library/share/gdal',
                           Path(prefix) / 'share/gdal'))
    candidates.append(_package_data('pyogrio'))
    if os.environ.get('GDAL_DATA'):
        candidates.append(Path(os.environ['GDAL_DATA']))
    for candidate in candidates:
        if candidate is not None and (candidate / 'gdalvrt.xsd').is_file():
            directory = str(candidate.resolve())
            os.environ['GDAL_DATA'] = directory
            return directory
    return None


def gdal_resources():
    """Describe the resource location used by this process and its children."""
    directory = os.environ.get('GDAL_DATA')
    schema = Path(directory) / 'gdalvrt.xsd' if directory else None
    return {'directory': directory,
            'schema_found': schema is not None and schema.is_file()}
