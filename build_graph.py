"""Build the user-agnostic global trajectory flow map from the sequence data"""

import os

from tqdm import tqdm, trange
import networkx as nx
import numpy as np
from scipy.sparse import save_npz
import pandas as pd
import argparse
from collections import Counter


def build_global_POI_checkin_graph(df):
    G = nx.DiGraph()

    # Add nodes by place_id
    # Doing this preserves ordering of place_id as node ids
    for place_id in trange(
        df["place_id"].max() + 1, desc="Add nodes to graph", ncols=80
    ):
        G.add_node(place_id, checkin_cnt=0, poi_catid=-1, latitude=0.0, longitude=0.0)

    for place_id, rows in tqdm(
        df.groupby("place_id"), desc="Add node features", ncols=80
    ):
        checkin_cnt = len(rows)
        G.nodes[place_id]["checkin_cnt"] = checkin_cnt
        G.nodes[place_id]["poi_catid"] = rows.iloc[0]["category"]
        G.nodes[place_id]["latitude"] = rows.iloc[0]["place_lat"]
        G.nodes[place_id]["longitude"] = rows.iloc[0]["place_lon"]

    # Add edges by user check-in sequences
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df = df.sort_values(by=["user_id", "arrival_time"])

    previous_poi_ids = df["place_id"].values[:-1]
    next_poi_ids = df["place_id"].values[1:]
    same_user_mask = df["user_id"].values[:-1] == df["user_id"].values[1:]
    edge_counter = Counter(
        zip(previous_poi_ids[same_user_mask], next_poi_ids[same_user_mask])
    )
    for (src_poi, dst_poi), weight in tqdm(
        edge_counter.items(), desc="Add edges", ncols=80
    ):
        G.add_edge(src_poi, dst_poi, weight=weight)

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix as a sparse file (avoid creating a dense matrix)
    nodelist = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # Save as compressed .npz preserving sparsity and ordering of nodelist
    save_npz(os.path.join(dst_dir, "graph_A.npz"), A)

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, "graph_X.csv"), "w") as f:
        print("place_id,checkin_cnt,poi_catid,latitude,longitude", file=f)
        for each in tqdm(nodes_data, desc="Save graph nodes", ncols=80):
            place_id = each[0]
            checkin_cnt = each[1]["checkin_cnt"]
            poi_catid = each[1]["poi_catid"]
            latitude = each[1]["latitude"]
            longitude = each[1]["longitude"]
            print(
                f"{place_id},{checkin_cnt},{poi_catid},{latitude},{longitude}",
                file=f,
            )


def load_graph_node_features(
    path,
    feature1="checkin_cnt",
    feature2="poi_catid",
    feature3="latitude",
    feature4="longitude",
):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str)
    args = parser.parse_args()

    dst_dir = f"dataset/{args.city}"
    data_path = f"/data/private/maria_data/data/trajfm_veraset_splits/veraset/Visits/{args.city}/whole_veraset_processed.parquet"

    # Build POI checkin trajectory graph
    train_df = pd.read_parquet(data_path)
    print("Build global POI checkin graph -----------------------------------")
    G = build_global_POI_checkin_graph(train_df)

    # Save graph to disk
    save_graph_to_csv(G, dst_dir=dst_dir)
