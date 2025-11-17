import pandas as pd
import wandb
import logging
import os
from pathlib import Path
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import (
    GCN,
    NodeAttnMap,
    UserEmbeddings,
    Time2Vec,
    CategoryEmbeddings,
    FuseEmbeddings,
    TransformerModel,
)
from param_parser import parameter_parser
from utils import (
    increment_path,
    calculate_laplacian_matrix,
    maksed_mse_loss,
)


def set_seed():
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If running on CUDA, seed all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """DataLoader worker seeding: ensures each worker has a distinct deterministic seed"""
    # each worker gets seed = base_seed + worker_id
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def datetime_to_float(dt, min_time, max_time):
    """
    Normalize the datetime object to a float in the range [0, 1].
    """
    if max_time == min_time:
        return 0.0  # Avoid division by zero if all timestamps are the same
    return (dt - min_time) / (max_time - min_time)


def train(args):
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        # Disable flash/memory-efficient attention kernels; fall back to math for stability
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    step = 0
    best_loss = float("inf")
    args.save_dir = increment_path(
        Path(args.project) / args.name, exist_ok=args.exist_ok, sep="-"
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(args.save_dir, f"log_training.txt"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.getLogger("matplotlib.font_manager").disabled = True

    # %% ====================== Load data ======================
    data_path = f"/data/private/maria_data/data/trajfm_veraset_splits/veraset/Visits/{args.city}/whole_veraset_processed.parquet"

    # Read check-in train data
    check_ins = pd.read_parquet(data_path)
    check_ins["arrival_time"] = pd.to_datetime(check_ins["arrival_time"])
    check_ins = check_ins.sort_values(by=["user_id", "arrival_time"])

    num_train = int(len(check_ins) * args.train_ratio)
    train_df = check_ins.iloc[:num_train].reset_index(drop=True)
    val_df = check_ins.iloc[num_train:].reset_index(drop=True)

    MIN_TIME = check_ins["arrival_time"].min()
    MAX_TIME = check_ins["arrival_time"].max()

    # Build POI graph (built from train_df)
    print("Loading POI graph...")
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(
        args.data_node_feats, args.feature1, args.feature2, args.feature3, args.feature4
    )
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}."
    )
    logging.info(
        f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency)."
    )
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info("One-hot encoding poi categories id")
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(
        list(map(lambda x: [x], cat_list))
    ).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1 : num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1 :] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")

    # Print how sparse raw_A is
    total_elements = raw_A.shape[0] * raw_A.shape[1]
    nonzero_elements = raw_A.count_nonzero()
    logging.info(
        f"Adjacency matrix sparsity: {nonzero_elements}/{total_elements} "
        f"({nonzero_elements / total_elements:.6f}) non-zero elements."
    )

    # Normalization
    print("Laplician matrix...")
    A = calculate_laplacian_matrix(raw_A)

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)

    # Cat id to index
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    poi_idx2cat_idx_dict = {
        poi_id: cat_id2idx_dict[cat_id]
        for poi_id, cat_id in nodes_df[["place_id", args.feature2]].values
    }

    # User id to index
    user_ids = [str(each) for each in list(set(train_df["user_id"].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            seq_len = 20
            for user, user_df in train_df.groupby("user_id"):
                for i in range(0, len(user_df), seq_len):
                    traj_df = user_df.iloc[i : i + seq_len]
                    traj_id = f"{user}_{i // seq_len}"

                    # Ger POIs idx in this trajectory
                    poi_idxs = traj_df["place_id"].to_list()
                    time_feature = datetime_to_float(
                        traj_df[args.time_feature], MIN_TIME, MAX_TIME
                    ).to_list()

                    # Construct input seq and label seq
                    input_seq = []
                    label_seq = []
                    for j in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[j], time_feature[j]))
                        label_seq.append((poi_idxs[j + 1], time_feature[j + 1]))

                    if len(input_seq) < args.short_traj_thres:
                        continue

                    self.traj_seqs.append(traj_id)
                    self.input_seqs.append(input_seq)
                    self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (
                self.traj_seqs[index],
                self.input_seqs[index],
                self.label_seqs[index],
            )

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            seq_len = 20
            for user, user_df in train_df.groupby("user_id"):
                for i in range(0, len(user_df), seq_len):
                    traj_df = user_df.iloc[i : i + seq_len]
                    traj_id = f"{user}_{i // seq_len}"

                    # Ger POIs idx in this trajectory
                    poi_idxs = traj_df["place_id"].to_list()
                    time_feature = datetime_to_float(
                        traj_df[args.time_feature], MIN_TIME, MAX_TIME
                    ).to_list()

                    # Construct input seq and label seq
                    input_seq = []
                    label_seq = []
                    for j in range(len(poi_idxs) - 1):
                        input_seq.append((poi_idxs[j], time_feature[j]))
                        label_seq.append((poi_idxs[j + 1], time_feature[j + 1]))

                    if len(input_seq) < args.short_traj_thres:
                        continue

                    self.traj_seqs.append(traj_id)
                    self.input_seqs.append(input_seq)
                    self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (
                self.traj_seqs[index],
                self.input_seqs[index],
                self.label_seqs[index],
            )

    # %% ====================== Define dataloader ======================
    print("Prepare dataloader...")
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=lambda x: x,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=lambda x: x,
        worker_init_fn=worker_init_fn,
    )

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    # Convert X (always dense) to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    # Handle adjacency: calculate_laplacian_matrix may return a scipy sparse CSR or dense ndarray
    A_sparse_torch = None

    # If A is a scipy sparse matrix, convert to torch sparse_coo_tensor for GCN
    if sp.issparse(A):
        A_csr = A.tocsr()
        A_coo = A_csr.tocoo()
        indices = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
        values = torch.FloatTensor(A_coo.data)
        A_sparse_torch = torch.sparse_coo_tensor(
            indices, values, size=A_coo.shape
        ).coalesce()
    else:
        # dense numpy array / matrix
        A_dense_np = np.array(A)
        # create sparse torch tensor as well for GCN (optional)
        A_coo = sp.coo_matrix(A_dense_np)
        indices = torch.LongTensor(np.vstack((A_coo.row, A_coo.col)))
        values = torch.FloatTensor(A_coo.data)
        A_sparse_torch = torch.sparse_coo_tensor(
            indices, values, size=A_coo.shape
        ).coalesce()

    # Move tensors to device and set dtype
    X = X.to(device=args.device, dtype=torch.float)
    A_sparse_torch = A_sparse_torch.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GCN(
        ninput=args.gcn_nfeat,
        nhid=args.gcn_nhid,
        noutput=args.poi_embed_dim,
        dropout=args.gcn_dropout,
    )

    # Node Attn Model
    node_attn_model = NodeAttnMap(
        in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False
    )

    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = Time2Vec("sin", out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

    # %% Model6: Sequence model
    args.seq_input_embed = (
        args.poi_embed_dim
        + args.user_embed_dim
        + args.time_embed_dim
        + args.cat_embed_dim
    )
    seq_model = TransformerModel(
        num_pois,
        num_cats,
        args.seq_input_embed,
        args.transformer_nhead,
        args.transformer_nhid,
        args.transformer_nlayers,
        dropout=args.transformer_dropout,
    )

    # Define overall loss and optimizer
    trainable_params = (
        list(poi_embed_model.parameters())
        + list(node_attn_model.parameters())
        + list(user_embed_model.parameters())
        + list(time_embed_model.parameters())
        + list(cat_embed_model.parameters())
        + list(embed_fuse_model1.parameters())
        + list(embed_fuse_model2.parameters())
        + list(seq_model.parameters())
    )

    optimizer = optim.Adam(
        params=trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

        # User to embedding
        user_id = traj_id.split("_")[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(
                    device=args.device
                )
            )
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # %% Loop epoch

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 25} Epoch: {epoch+1} / {args.epochs} {'*' * 25}")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()

        # Mask will be (re)built per-batch using actual seq_len
        src_mask = None
        # Loop batch
        pbar = tqdm(train_loader, desc="Training", ncols=80)
        for batch in pbar:
            # Build inputs and masks from the batch's actual sequence length

            # Assert no nan in batch
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                assert (
                    not any(pd.isna(input_seq))
                    and not any(pd.isna(label_seq))
                    and not any(pd.isna(input_seq_time))
                    and not any(pd.isna(label_seq_time))
                ), f"NaN detected in batch data: {sample[0]}"

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A_sparse_torch)

            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                # input_seq_embed shape: [seq_len, embed_dim]
                input_seq_embed = torch.stack(
                    input_traj_to_embeddings(sample, poi_embeddings), dim=0
                )
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            # batch_padded shape: [batch, seq, embed_dim]
            batch_padded = pad_sequence(
                batch_seq_embeds, batch_first=True, padding_value=-1
            )
            label_padded_poi = pad_sequence(
                batch_seq_labels_poi, batch_first=True, padding_value=-1
            )
            label_padded_time = pad_sequence(
                batch_seq_labels_time, batch_first=True, padding_value=-1
            )
            label_padded_cat = pad_sequence(
                batch_seq_labels_cat, batch_first=True, padding_value=-1
            )

            # Build key padding mask: True where all features are padding (-1)
            pad_mask = (batch_padded == -1).all(dim=2)  # [batch, seq]

            # Skip batch if everything is padded across all targets (avoids NaN losses)
            has_poi = (label_padded_poi != -1).any().item()
            has_time = (
                ((label_padded_time != -1) & torch.isfinite(label_padded_time))
                .any()
                .item()
            )
            has_cat = (label_padded_cat != -1).any().item()
            if not (has_poi or has_time or has_cat):
                # Nothing to learn in this batch
                continue

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            # Zero padded positions to avoid injecting padding values into FFN/residuals
            if pad_mask.any():
                x[pad_mask.unsqueeze(-1).expand_as(x)] = 0.0
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)

            seq_len = x.size(1)
            src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, args.device, dtype=torch.bool
            )
            key_padding_mask = pad_mask.to(args.device)  # [batch, seq]

            if torch.isnan(x).any():
                raise ValueError("NaN detected in input x (transposed).")
            if torch.isnan(y_poi).any():
                raise ValueError("NaN detected in label y_poi.")
            if torch.isnan(y_time).any():
                raise ValueError("NaN detected in label y_time.")
            if torch.isnan(y_cat).any():
                raise ValueError("NaN detected in label y_cat.")
            if torch.isnan(src_mask).any():
                raise ValueError("NaN detected in src_mask.")

            y_pred_poi, y_pred_time, y_pred_cat = seq_model(
                x, src_mask, key_padding_mask
            )

            # Assert no nan in predictions
            if torch.isnan(y_pred_poi).any():
                raise ValueError("NaN detected in prediction y_pred_poi.")
            if torch.isnan(y_pred_time).any():
                raise ValueError("NaN detected in prediction y_pred_time.")
            if torch.isnan(y_pred_cat).any():
                raise ValueError("NaN detected in prediction y_pred_cat.")

            valid_positions = ~key_padding_mask

            flat_valid_poi = (valid_positions & (y_poi != -1)).view(-1)
            flat_valid_time = (
                valid_positions & (y_time != -1) & torch.isfinite(y_time)
            ).view(-1)
            flat_valid_cat = (valid_positions & (y_cat != -1)).view(-1)

            flat_pred_poi = y_pred_poi.view(-1, y_pred_poi.size(-1))
            flat_y_poi = y_poi.view(-1)
            flat_pred_time = y_pred_time.view(-1)
            flat_y_time = y_time.view(-1)
            flat_pred_cat = y_pred_cat.view(-1, y_pred_cat.size(-1))
            flat_y_cat = y_cat.view(-1)

            if not flat_valid_poi.any():
                # Should not happen because we skip empty batches above, but guard anyway
                continue

            loss_poi = criterion_poi(
                flat_pred_poi[flat_valid_poi], flat_y_poi[flat_valid_poi]
            )
            if flat_valid_time.any():
                loss_time = criterion_time(
                    flat_pred_time[flat_valid_time], flat_y_time[flat_valid_time]
                )
            else:
                loss_time = flat_pred_time.new_tensor(0.0)

            if flat_valid_cat.any():
                loss_cat = criterion_cat(
                    flat_pred_cat[flat_valid_cat], flat_y_cat[flat_valid_cat]
                )
            else:
                loss_cat = flat_pred_cat.new_tensor(0.0)

            # Final loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as err:
                if "ScaledDotProduct" in str(err):
                    logging.error("ScaledDotProduct attention backward failed: %s", err)
                    logging.error(
                        "x abs max %.4e | abs mean %.4e",
                        x.abs().max().item(),
                        x.abs().mean().item(),
                    )
                    logging.error(
                        "y_pred_poi abs max %.4e",
                        y_pred_poi.abs().max().item(),
                    )
                    logging.error(
                        "src_mask finite=%s min=%s max=%s",
                        (
                            torch.isfinite(src_mask).all().item()
                            if src_mask is not None
                            else True
                        ),
                        src_mask.min().item() if src_mask is not None else 0.0,
                        src_mask.max().item() if src_mask is not None else 0.0,
                    )
                    logging.error(
                        "key_padding_mask true_count=%d",
                        int(key_padding_mask.sum().item()),
                    )
                raise
            # Additional clipping by value as an extra safety guard
            torch.nn.utils.clip_grad_value_(trainable_params, clip_value=5.0)
            # Clip gradients to avoid rare spikes causing NaNs
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            # Check gradients for debugging
            params_with_nan_grad = []
            for name, param in seq_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        params_with_nan_grad.append(name)
            if params_with_nan_grad:
                raise ValueError(
                    f"After backward: NaN detected in gradient of parameter: {params_with_nan_grad}"
                )

            optimizer.step()

            step += 1
            run.log({"train/loss": loss.item()}, step=step)
            pbar.set_postfix({"loss": loss.item()})
        pbar.close()
        # train end --------------------------------------------------------------------------------------------------------

        # If no val data, directly save embeddings and skip val
        if args.train_ratio + 1e-5 >= 1.0:
            if not torch.isnan(loss):
                # Save epoch embeddings
                poi_embeddings = poi_embed_model(X, A_sparse_torch).detach().cpu()
                embeddings = {
                    idx: poi_embeddings[idx] for idx in range(len(poi_embeddings))
                }
                torch.save(embeddings, f"output/{run.name}.pt")
            else:
                raise ValueError(
                    "NaN detected in training loss, aborting saving embeddings."
                )
            continue

        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()

        for batch in tqdm(val_loader, desc="Validation", ncols=80):
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A_sparse_torch)

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(
                    input_traj_to_embeddings(sample, poi_embeddings)
                )
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(
                batch_seq_embeds, batch_first=True, padding_value=-1
            )
            label_padded_poi = pad_sequence(
                batch_seq_labels_poi, batch_first=True, padding_value=-1
            )
            label_padded_time = pad_sequence(
                batch_seq_labels_time, batch_first=True, padding_value=-1
            )
            label_padded_cat = pad_sequence(
                batch_seq_labels_cat, batch_first=True, padding_value=-1
            )

            # Build key padding mask
            pad_mask = (batch_padded == -1).all(dim=2)  # [batch, seq]

            # Skip batch if everything is padded across all targets
            has_poi = (label_padded_poi != -1).any().item()
            has_time = (
                ((label_padded_time != -1) & torch.isfinite(label_padded_time))
                .any()
                .item()
            )
            has_cat = (label_padded_cat != -1).any().item()
            if not (has_poi or has_time or has_cat):
                continue

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            if pad_mask.any():
                x[pad_mask.unsqueeze(-1).expand_as(x)] = 0.0
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            seq_len = x.size(1)
            src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, args.device, dtype=torch.bool
            )
            key_padding_mask = pad_mask.to(args.device)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(
                x, src_mask, key_padding_mask
            )

            # Calculate loss
            valid_positions = ~key_padding_mask
            flat_valid_poi = (valid_positions & (y_poi != -1)).view(-1)
            flat_valid_time = (
                valid_positions & (y_time != -1) & torch.isfinite(y_time)
            ).view(-1)
            flat_valid_cat = (valid_positions & (y_cat != -1)).view(-1)

            flat_pred_poi = y_pred_poi.view(-1, y_pred_poi.size(-1))
            flat_y_poi = y_poi.view(-1)
            flat_pred_time = y_pred_time.view(-1)
            flat_y_time = y_time.view(-1)
            flat_pred_cat = y_pred_cat.view(-1, y_pred_cat.size(-1))
            flat_y_cat = y_cat.view(-1)

            if flat_valid_poi.any():
                loss_poi = criterion_poi(
                    flat_pred_poi[flat_valid_poi], flat_y_poi[flat_valid_poi]
                )
            else:
                loss_poi = flat_pred_poi.new_tensor(0.0)

            if flat_valid_time.any():
                loss_time = criterion_time(
                    flat_pred_time[flat_valid_time], flat_y_time[flat_valid_time]
                )
            else:
                loss_time = flat_pred_time.new_tensor(0.0)

            if flat_valid_cat.any():
                loss_cat = criterion_cat(
                    flat_pred_cat[flat_valid_cat], flat_y_cat[flat_valid_cat]
                )
            else:
                loss_cat = flat_pred_cat.new_tensor(0.0)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

            run.log({"val/loss": loss.item()}, step=step)
        # valid end --------------------------------------------------------------------------------------------------------

        # Save best epoch embeddings
        if loss < best_loss:
            best_loss = loss

            # Save poi embeddings
            poi_embeddings = poi_embed_model(X, A_sparse_torch).detach().cpu()
            embeddings = {
                idx: poi_embeddings[idx] for idx in range(len(poi_embeddings))
            }
            torch.save(embeddings, f"output/{run.name}.pt")


if __name__ == "__main__":
    args = parameter_parser()
    args.feature1 = "checkin_cnt"
    args.feature2 = "poi_catid"
    args.feature3 = "latitude"
    args.feature4 = "longitude"

    set_seed()

    args.data_adj_mtx = f"dataset/{args.city}/graph_A.npz"
    args.data_node_feats = f"dataset/{args.city}/graph_X.csv"

    run = wandb.init(
        project="me-poi-baselines",
        name=f"getnext_{args.city}_lr{args.lr}",
        config=vars(args),
        # mode="offline",
    )
    train(args)
    run.finish()
