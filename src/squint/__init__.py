from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("squint")
except PackageNotFoundError:  # imported from a source tree without being installed
    __version__ = "unknown"
