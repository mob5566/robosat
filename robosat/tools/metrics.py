import sys
import argparse

import geojson

from tqdm import tqdm
import shapely.geometry

from robosat.spatial.core import make_index, union, project_ea, project_wgs_el, project_el_wgs
from robosat.graph.core import UndirectedGraph


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
    with open(args.predicts) as fp:
        collection = geojson.load(fp)

    predicts = [shapely.geometry.shape(feature["geometry"]) for feature in collection["features"]]
    del collection

    with open(args.labels) as fp:
        collection = geojson.load(fp)

    labels = [shapely.geometry.shape(feature["geometry"]) for feature in collection["features"]]
    del collection

    pidx = make_index(predicts)
    lidx = make_index(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    true_pos = []
    false_pos = []
    false_neg = []

    for i, predict in enumerate(tqdm(predicts, desc="Scanning predictions", unit="shapes", ascii=True)):

        feature = geojson.Feature(geometry=shapely.geometry.mapping(predict))
        nearest = [j for j in lidx.intersection(predict.bounds, objects=False)]

        matched = False
        for t in nearest:
            if predict.intersects(labels[t]):
                tp += 1
                true_pos.append(feature)
                matched = True
                break

        if not matched:
            false_pos.append(feature)
            fp += 1

    for i, label in enumerate(tqdm(labels, desc="Scanning labels", unit="shapes", ascii=True)):

        feature = geojson.Feature(geometry=shapely.geometry.mapping(label))
        nearest = [j for j in pidx.intersection(label.bounds, objects=False)]

        matched = False
        for t in nearest:
            if label.intersects(predicts[t]):
                matched = True
                break

        if not matched:
            false_neg.append(feature)
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'precision: {precision}, recall: {recall}, f1_score: {f1_score}')
    print(f'true pos: {tp}, false pos: {fp}, false neg: {fn}')

    def save_geojson(fname, features):
        collection = geojson.FeatureCollection(features)

        with open(fname, "w") as fp:
            geojson.dump(collection, fp)

    save_geojson(args.out + '_tp.geojson', true_pos)
    save_geojson(args.out + '_fp.geojson', false_pos)
    save_geojson(args.out + '_fn.geojson', false_neg)
