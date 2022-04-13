import argparse

import numpy as np

from PIL import Image
from tqdm import tqdm

from robosat.tiles import tiles_from_slippy_map
from robosat.config import load_config
from robosat.colors import Mapbox

from robosat.features.parking import ParkingHandler
from robosat.features.bridge import BridgeHandler
from robosat.features.landuse import LanduseHandler


# Register post-processing handlers here; they need to support a `apply(tile, mask)` function
# for handling one mask and a `save(path)` function for GeoJSON serialization to a file.
handlers = {
    "background": None,
    "bridge": BridgeHandler,
    "Veg": LanduseHandler,
    "Orchard": LanduseHandler,
    "Swamp": LanduseHandler,
    "Sandbar": LanduseHandler,
    "WaterBody": LanduseHandler,
    "Builtup": LanduseHandler,
    "Build": LanduseHandler,
}


def add_parser(subparser):
    parser = subparser.add_parser(
        "features",
        help="extracts simplified GeoJSON features from segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("masks", type=str, help="slippy map directory with segmentation masks")
    parser.add_argument("--from-palette", action='store_true', default=False, help="whether the values of masks are from palette")
    parser.add_argument("--type", type=str, required=True, choices=handlers.keys(), help="type of feature to extract")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")
    parser.add_argument("out", type=str, help="path to GeoJSON file to store features in")

    parser.set_defaults(func=main)


def main(args):
    dataset = load_config(args.dataset)

    labels = dataset["common"]["classes"]
    assert set(handlers.keys()).issuperset(set(labels)), "handlers have a class label"
    index = labels.index(args.type)
    color = Mapbox[dataset["common"]["colors"][index]].value

    handler = handlers[args.type]()

    tiles = list(tiles_from_slippy_map(args.masks))

    for tile, path in tqdm(tiles, ascii=True, unit="mask"):
        if args.from_palette:
            image = np.array(Image.open(path), dtype=np.uint8)
            mask = np.all(image == color, axis=-1).astype(np.uint8)
        else:
            image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
            mask = (image == index).astype(np.uint8)

        handler.apply(tile, mask)

    handler.save(args.out)
