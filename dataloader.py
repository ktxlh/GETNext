import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def load_graph_adj_mtx(path):
    """Load adjacency matrix.

    Supports sparse `.npz` files (produced by `scipy.sparse.save_npz`) and
    legacy dense CSVs. Returns a scipy CSR matrix when loading `.npz`, or a
    NumPy array when loading CSV.

    A.shape: (num_node, num_node), edge from row_index to col_index with weight
    """
    if str(path).endswith(".npz"):
        return load_npz(path)
    else:
        A = np.loadtxt(path, delimiter=",")
        return A


def load_graph_node_features(
    path,
    feature1="checkin_cnt",
    feature2="poi_catid_code",
    feature3="latitude",
    feature4="longitude",
):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X
