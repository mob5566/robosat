import matplotlib

import geojson

from tqdm import tqdm
import shapely.geometry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robosat.spatial.core import make_index, union, project_ea, project_wgs_el, project_el_wgs

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

def get_instance_metrics(predict_geojson, ground_truth_geojson):
    with open(predict_geojson) as fp:
        collection = geojson.load(fp)

    predicts = [shapely.geometry.shape(feature["geometry"]) for feature in collection["features"]]
    del collection

    with open(ground_truth_geojson) as fp:
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

        area = int(round(project_ea(predict).area))
        feature = geojson.Feature(geometry=shapely.geometry.mapping(predict), properties={"area": area})
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

        area = int(round(project_ea(label).area))
        feature = geojson.Feature(geometry=shapely.geometry.mapping(label), properties={"area": area})
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
    accuracy = tp / len(labels)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return ((tp, fp, fn, tn),
            (precision, recall, accuracy, f1_score),
            (true_pos, false_pos, false_neg))
