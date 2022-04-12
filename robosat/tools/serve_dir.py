import os
import io
import sys
import argparse
import pathlib

import numpy as np

import mercantile
import requests
from PIL import Image
from flask import Flask, send_file, render_template, abort

from robosat.tiles import fetch_image
from robosat.unet import UNet
from robosat.config import load_config
from robosat.colors import make_palette
from robosat.transforms import ConvertImageMode, ImageToTensor

"""
Simple tile server running a segmentation model on the fly.

Endpoints:
  /zoom/x/y.png  Segmentation mask PNG image for the corresponding tile

Note: proof of concept for quick visualization only; limitations:
  Needs to be single threaded, request runs prediction on the GPU (singleton); should be batch prediction
  Does not take surrounding tiles into account for prediction; border predictions do not have to match
  Downloads satellite images for each request; should request internal data or at least do some caching
"""

app = Flask(__name__)

session = None
tiles_dir = None
size = None


@app.route("/")
def index():
    return render_template("map.html", size=size)


@app.route("/<int:z>/<int:x>/<int:y>.png")
def tile(z, x, y):

    # Todo: predictor should take care of zoom levels
    if z != 15:
        abort(404)

    tile = mercantile.Tile(x, y, z)

    path = (tiles_dir / str(tile.z) / str(tile.x) / str(tile.y)).with_suffix('.png')

    if not path.is_file():
        abort(500)

    mask = Image.open(path)

    return send_png(mask)


@app.after_request
def after_request(response):
    header = response.headers
    header["Access-Control-Allow-Origin"] = "*"
    return response


def add_parser(subparser):
    parser = subparser.add_parser(
        "serve_dir",
        help="serves predicted masks with on-demand tileserver from"
             "a source directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("src_dir", type=str, help="path to the source directory")

    parser.add_argument("--tile_size", type=int, default=256, help="tile size for slippy map tiles")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host to serve on")
    parser.add_argument("--port", type=int, default=5000, help="port to serve on")

    parser.set_defaults(func=main)


def main(args):
    global size
    size = args.tile_size

    global session
    session = requests.Session()

    global tiles_dir
    tiles_dir = pathlib.Path(args.src_dir)

    app.run(host=args.host, port=args.port, threaded=False)


def send_png(image):
    output = io.BytesIO()
    image.save(output, format="png", optimize=True)
    output.seek(0)
    return send_file(output, mimetype="image/png")
