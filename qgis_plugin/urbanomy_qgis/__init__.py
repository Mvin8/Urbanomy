"""QGIS plugin entrypoint for Urbanomy."""


def classFactory(iface):  # noqa: N802 - QGIS plugin API
    """Load the Urbanomy plugin class."""
    from .plugin import UrbanomyQgisPlugin

    return UrbanomyQgisPlugin(iface)
