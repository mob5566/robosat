import sys
import argparse

import geojson

from robosat.graph.core import UndirectedGraph
from robosat.utils import get_instance_metrics

def add_parser(subparser):
    parser = subparser.add_parser(
        "metrics", help="calculate metrics from GeoJSON features", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("predicts", type=str, help="GeoJSON file to read prediction features from")
    parser.add_argument("labels", type=str, help="GeoJSON file to read label features from")
    parser.add_argument("--threshold", type=int, required=False, help="minimum distance to adjacent features, in m")
    parser.add_argument("out", type=str, help="path to GeoJSON to save merged features to")

    parser.set_defaults(func=main)


def main(args):
    ((tp, fp, fn, tn),
     (precision, recall, accuracy, f1_score),
     (true_pos, false_pos, false_neg)) = get_instance_metrics(args.predicts,
                                                              args.labels)

    print(f'precision: {precision}, recall: {recall}, f1_score: {f1_score}, accuracy: {accuracy}')
    print(f'true pos: {tp}, false pos: {fp}, false neg: {fn}')

    def save_geojson(fname, features):
        collection = geojson.FeatureCollection(features)

        with open(fname, "w") as fp:
            geojson.dump(collection, fp)

    save_geojson(args.out + '_tp.geojson', true_pos)
    save_geojson(args.out + '_fp.geojson', false_pos)
    save_geojson(args.out + '_fn.geojson', false_neg)
