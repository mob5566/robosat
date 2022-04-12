import matplotlib
import numpy as np

import geojson
from tqdm import tqdm
import shapely.geometry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robosat.spatial.core import make_index, union, project_ea, project_wgs_el, project_el_wgs
from robosat.datasets import SlippyMapTiles

def plot(out, history):
    plt.figure()

    n = max(map(len, history.values()))

    plt.grid()

    for values in history.values():
        plt.plot(values)

    plt.xlabel("epoch")
    plt.legend(list(history))

    plt.savefig(out, format="png")
    plt.close()

def get_instance_metrics(predict_geojson, ground_truth_geojson, threshold=0.5):
    with open(predict_geojson) as fp:
        collection = geojson.load(fp)

    predicts = list(filter(lambda shape: shape.area > 0,
                           [shapely.geometry.shape(feature["geometry"]).buffer(0)
                            for feature in collection["features"]]))
    del collection

    with open(ground_truth_geojson) as fp:
        collection = geojson.load(fp)

    labels = list(filter(lambda shape: shape.area > 0,
                         [shapely.geometry.shape(feature["geometry"]).buffer(0)
                          for feature in collection["features"]]))
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

    def hit(poly1, poly2):
        return (poly1.intersection(poly2).area /
                (poly1.union(poly2).area + 1e-8)) >= threshold

    for i, predict in enumerate(tqdm(predicts, desc="Scanning predictions", unit="shapes", ascii=True)):

        area = int(round(project_ea(predict).area))
        feature = geojson.Feature(geometry=shapely.geometry.mapping(predict), properties={"area": area})
        nearest = [j for j in lidx.intersection(predict.bounds, objects=False)]

        matched = False
        for t in nearest:
            if hit(predict, labels[t]):
                tp += 1
                true_pos.append(feature)
                matched = True
                break

        if not matched:
            false_pos.append(feature)
            fp += 1

    for i, label in enumerate(tqdm(labels, desc="Scanning labels", unit="shapes", ascii=True)):

        area = int(round(project_ea(label).area))
        feature = geojson.Feature(geometry=shapely.geometry.mapping(label), properties={"area": area})
        nearest = [j for j in pidx.intersection(label.bounds, objects=False)]

        matched = False
        for t in nearest:
            if hit(label, predicts[t]):
                matched = True
                break

        if not matched:
            false_neg.append(feature)
            fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = tp / len(labels)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return ((tp, fp, fn, tn),
            (precision, recall, accuracy, f1_score),
            (true_pos, false_pos, false_neg))

def get_pixel_metrics(predicted_dir, ground_truth_dir):
    pt_dataset = SlippyMapTiles(predicted_dir)
    gt_dataset = SlippyMapTiles(ground_truth_dir)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pt, gt in tqdm(zip(pt_dataset, gt_dataset)):
        pt = np.array(pt[0])
        gt = np.array(gt[0])

        tp += np.sum(np.logical_and(pt == 1, gt == 1))
        fp += np.sum(np.logical_and(pt == 1, gt == 0))
        fn += np.sum(np.logical_and(pt == 0, gt == 1))
        tn += np.sum(np.logical_and(pt == 0, gt == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    miou = tp / (tp + fp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return ((tp, fp, fn, tn),
            (precision, recall, accuracy, miou, f1_score))
