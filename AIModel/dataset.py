"""
PyTorch Dataset wrapper + subject-wise (block-level) train/val/test split.
Splitting by volunteer ("groups"), not by window, is what prevents leakage:
a given volunteer's windows only ever appear in one of train/val/test.

Now carries two parallel arrays per window: the raw sequence (X_seq) and
the engineered scalar features (X_feat).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

import config as cfg


class WindowDataset(Dataset):
    def __init__(self, X_seq, X_feat, y):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.X_feat = torch.from_numpy(X_feat).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]


def subject_wise_split(X_seq, X_feat, y, groups, test_fraction=cfg.TEST_FRACTION,
                        val_fraction=cfg.VAL_FRACTION, seed=cfg.RANDOM_SEED):
    """
    Splits by volunteer (group), never by individual window, so no
    volunteer appears in more than one split.
    Returns dict with 'train', 'val', 'test' -> (X_seq, X_feat, y, groups) tuples.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(X_seq, y, groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(X_seq[trainval_idx], y[trainval_idx], groups[trainval_idx])
    )
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    def pack(idx):
        return X_seq[idx], X_feat[idx], y[idx], groups[idx]

    splits = {"train": pack(train_idx), "val": pack(val_idx), "test": pack(test_idx)}

    train_v = set(splits["train"][3])
    val_v = set(splits["val"][3])
    test_v = set(splits["test"][3])
    assert not (train_v & val_v), "Leakage: volunteer in both train and val"
    assert not (train_v & test_v), "Leakage: volunteer in both train and test"
    assert not (val_v & test_v), "Leakage: volunteer in both val and test"

    return splits


def compute_global_stats(X):
    """
    Global fixed normalization computed on the TRAIN split only, applied
    afterwards to train/val/test/deployment alike. Works for both the
    3D sequence array [N,T,C] and the 2D scalar-feature array [N,F].
    """
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_global_stats(X, mean, std):
    return (X - mean) / std