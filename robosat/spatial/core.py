import functools

import pyproj
import shapely.ops

from rtree.index import Index, Property


def project(shape, source, target):
    """Projects a geometry from one coordinate system into another.

    Args:
      shape: the geometry to project.
      source: the source EPSG spatial reference system identifier.
      target: the target EPSG spatial reference system identifier.

    Returns:
      The projected geometry in the target coordinate system.
    """

    transformer = pyproj.Transformer.from_crs(source, target)
    return shapely.ops.transform(transformer.transform, shape)


def union(shapes):
    """Returns the union of all shapes.

    Args:
      shapes: the geometries to merge into one.

    Returns:
      The union of all shapes as one shape.
    """

    assert shapes

    def fn(lhs, rhs):
        return lhs.union(rhs)

    return functools.reduce(fn, shapes)

ea_transformer = pyproj.Transformer.from_crs("epsg:4326", "esri:54009", always_xy=True) 
wgs_ellipsoid_transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3395", always_xy=True)
ellipsoid_wgs_transformer = pyproj.Transformer.from_crs("epsg:3395", "epsg:4326", always_xy=True)

def project_ea(shape):
    return shapely.ops.transform(ea_transformer.transform, shape)

def project_wgs_el(shape):
    return shapely.ops.transform(wgs_ellipsoid_transformer.transform, shape)

def project_el_wgs(shape):
    return shapely.ops.transform(ellipsoid_wgs_transformer.transform, shape)


def iou(lhs, rhs):
    """Calculates intersection over union metric between two shapes..

    Args:
      lhs: first shape for IoU calculation.
      rhs: second shape for IoU calculation.

    Returns:
      IoU metric in range [0, 1]
    """

    # equal-area projection for comparing shape areas
    lhs = project_ea(lhs)
    rhs = project_ea(rhs)

    intersection = lhs.intersection(rhs)
    union = lhs.union(rhs)

    rv = intersection.area / union.area
    assert 0 <= rv <= 1

    return rv


def make_index(shapes):
    """Creates an index for fast and efficient spatial queries.

    Args:
      shapes: shapely shapes to bulk-insert bounding boxes for into the spatial index.

    Returns:
      The spatial index created from the shape's bounding boxes.
    """

    # Todo: benchmark these for our use-cases
    prop = Property()
    prop.dimension = 2
    prop.leaf_capacity = 1000
    prop.fill_factor = 0.9

    def bounded():
        for i, shape in enumerate(shapes):
            yield (i, shape.bounds, None)

    return Index(bounded(), properties=prop)
