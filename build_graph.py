"""Build the user-agnostic global trajectory flow map from the sequence data"""

import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def build_global_POI_checkin_graph(df):
    G = nx.DiGraph()
    for user_id, user_df in df.groupby("user_id"):

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row["place_id"]
            if node not in G.nodes():
                G.add_node(
                    row["place_id"],
                    checkin_cnt=1,
                    poi_catid=row["category"],
                    latitude=row["place_lat"],
                    longitude=row["place_lon"],
                )
            else:
                G.nodes[node]["checkin_cnt"] += 1

        # Add edges (Check-in seq)
        previous_poi_id = -1
        for i, row in user_df.iterrows():
            if previous_poi_id == -1:
                previous_poi_id = row["place_id"]

            poi_id = row["place_id"]

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]["weight"] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(os.path.join(dst_dir, "graph_A.csv"), A.todense(), delimiter=",")

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, "graph_X.csv"), "w") as f:
        print("place_id,checkin_cnt,poi_catid,latitude,longitude", file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]["checkin_cnt"]
            poi_catid = each[1]["poi_catid"]
            latitude = each[1]["latitude"]
            longitude = each[1]["longitude"]
            print(
                f"{node_name},{checkin_cnt}," f"{poi_catid}," f"{latitude},{longitude}",
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
