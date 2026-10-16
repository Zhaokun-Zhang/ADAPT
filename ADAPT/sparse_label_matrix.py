"""
Sparse label matrix for llm_joined.

Stores labeled (Lid, Rid) -> label pairs in a dictionary instead of a dense
m by n NumPy array, which keeps memory use small.

Memory:
- dense matrix: m * n * 1 byte (for example 10k x 100k = 1GB)
- sparse matrix: about 24 bytes * #labeled_pairs (for example 10k pairs = 240KB)
"""

from typing import Dict, Set, Tuple, Union
from collections import defaultdict
import numpy as np


class SparseLabelMatrix:
    """
    Sparse label matrix. Only labeled (Lid, Rid) pairs are stored.

    Storage:
    - _labels: Dict[(int, int), int]  labeled pairs only
    - _pos_pairs: Set[(int, int)]     positive index (label=1)
    - _neg_pairs: Set[(int, int)]     negative index (label=0)
    - _row_index: Dict[int, Set[int]] row index for row-slice queries

    Supported operations:
    - mat[Lid, Rid]: point query, returns -1/0/1
    - mat[Lid, Rid] = label: point update
    - mat[Lid][Rid_arr]: row-slice query, returns np.ndarray
    - mat[Lid_arr, Rid_arr] = label_arr: batch update
    - mat.get_positive_pairs(): all positive pairs, in place of np.where(mat == 1)
    """

    def __init__(self, m: int, n: int):
        """
        Args:
            m: number of rows in the left table
            n: number of rows in the right table
        """
        self.shape = (m, n)
        self.m, self.n = m, n

        # Labeled pairs only.
        self._labels: Dict[Tuple[int, int], int] = {}

        # Secondary indexes for full scans.
        self._pos_pairs: Set[Tuple[int, int]] = set()  # label=1
        self._neg_pairs: Set[Tuple[int, int]] = set()  # label=0

        # Row index for mat[Lid][Rid_arr].
        self._row_index: Dict[int, Set[int]] = defaultdict(set)  # Lid -> {Rid...}

    # ========== Point access ==========

    def __getitem__(self, key):
        """
        Three access modes:
        1. mat[Lid, Rid] -> int (-1/0/1)
        2. mat[Lid][Rid_arr] -> np.ndarray, a batch query through RowView
        3. mat[Lid] -> RowView, which then accepts [Rid_arr]
        """
        if isinstance(key, tuple) and len(key) == 2:
            # mat[Lid, Rid]
            Lid, Rid = int(key[0]), int(key[1])
            return self._labels.get((Lid, Rid), -1)
        elif isinstance(key, (int, np.integer)):
            # mat[Lid] -> RowView
            return RowView(self, int(key))
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key, value):
        """
        Two assignment modes:
        1. mat[Lid, Rid] = label
        2. mat[Lid_arr, Rid_arr] = label_arr (NumPy fancy indexing)
        """
        if isinstance(key, tuple) and len(key) == 2:
            Lid, Rid = key
            if isinstance(Lid, (int, np.integer)) and isinstance(Rid, (int, np.integer)):
                self._set_single(int(Lid), int(Rid), int(value))
            else:
                # mat[Lid_arr, Rid_arr] = label_arr
                Lid_arr = np.atleast_1d(Lid)
                Rid_arr = np.atleast_1d(Rid)
                value_arr = np.atleast_1d(value)
                if np.isscalar(value):
                    # Broadcast mat[Lid_arr, Rid_arr] = scalar.
                    value_arr = np.full(len(Lid_arr), value, dtype=np.int8)
                for L, R, v in zip(Lid_arr, Rid_arr, value_arr):
                    self._set_single(int(L), int(R), int(v))
        else:
            raise TypeError(f"Unsupported assignment key: {key}")

    def _set_single(self, Lid: int, Rid: int, label: int):
        """Set one entry and update the indexes."""
        pair = (Lid, Rid)
        old_label = self._labels.get(pair, -1)

        # Drop the pair from its previous index.
        if old_label == 1:
            self._pos_pairs.discard(pair)
        elif old_label == 0:
            self._neg_pairs.discard(pair)

        # Update the primary store.
        if label == -1:
            self._labels.pop(pair, None)
            self._row_index[Lid].discard(Rid)
        else:
            self._labels[pair] = label
            self._row_index[Lid].add(Rid)
            if label == 1:
                self._pos_pairs.add(pair)
            elif label == 0:
                self._neg_pairs.add(pair)

    def get(self, Lid: int, Rid: int, default=-1) -> int:
        """Dictionary-style access: mat.get(Lid, Rid)."""
        return self._labels.get((int(Lid), int(Rid)), default)

    # ========== Row slices ==========

    def get_row_labels(self, Lid: int, Rid_arr: np.ndarray) -> np.ndarray:
        """
        Batch query for mat[Lid][Rid_arr].
        Returns an array of shape (len(Rid_arr),) filled with -1 by default.
        """
        Lid = int(Lid)
        result = np.full(len(Rid_arr), -1, dtype=np.int8)
        labeled_Rids = self._row_index.get(Lid, set())

        if labeled_Rids:
            # Look up only Rids that are already labeled.
            for i, Rid in enumerate(Rid_arr):
                Rid = int(Rid)
                if Rid in labeled_Rids:
                    result[i] = self._labels[(Lid, Rid)]
        return result

    # ========== Full scans ==========

    def get_positive_pairs(self) -> np.ndarray:
        """All positive pairs, in place of np.where(mat == 1). Shape (K, 2)."""
        if not self._pos_pairs:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(sorted(self._pos_pairs), dtype=np.int64)

    def get_negative_pairs(self) -> np.ndarray:
        """All negative pairs, in place of np.where(mat == 0). Shape (N, 2)."""
        if not self._neg_pairs:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(sorted(self._neg_pairs), dtype=np.int64)

    def get_labeled_pairs(self) -> np.ndarray:
        """All labeled pairs, in place of np.where(mat >= 0). Shape (L, 2)."""
        if not self._labels:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(sorted(self._labels.keys()), dtype=np.int64)

    def count_labeled(self) -> int:
        """In place of (mat >= 0).sum()."""
        return len(self._labels)

    def count_positive(self) -> int:
        """In place of (mat == 1).sum()."""
        return len(self._pos_pairs)

    def count_negative(self) -> int:
        """In place of (mat == 0).sum()."""
        return len(self._neg_pairs)

    # ========== Compatibility ==========

    def astype(self, dtype):
        """Accept glbinfo.llm_joined.astype(np.int8) and return this matrix."""
        return self

    def to_dense(self) -> np.ndarray:
        """Convert to a dense matrix for debugging or older caches."""
        mat = np.full(self.shape, -1, dtype=np.int8)
        for (Lid, Rid), label in self._labels.items():
            mat[Lid, Rid] = label
        return mat

    @staticmethod
    def from_dense(dense_mat: np.ndarray) -> 'SparseLabelMatrix':
        """Build a sparse matrix from a dense one, for an existing cache."""
        m, n = dense_mat.shape
        sparse = SparseLabelMatrix(m, n)
        labeled = np.argwhere(dense_mat >= 0)
        for Lid, Rid in labeled:
            sparse._set_single(int(Lid), int(Rid), int(dense_mat[Lid, Rid]))
        return sparse

    # ========== Serialization ==========

    def to_dict(self) -> dict:
        """Serialize to a pickle-friendly dictionary."""
        return {
            'shape': self.shape,
            'labels': dict(self._labels),  # Dict[(Lid, Rid), label]
        }

    @staticmethod
    def from_dict(data: dict) -> 'SparseLabelMatrix':
        """Deserialize from a dictionary."""
        sparse = SparseLabelMatrix(*data['shape'])
        for (Lid, Rid), label in data['labels'].items():
            sparse._set_single(Lid, Rid, label)
        return sparse

    def __repr__(self):
        return (f"SparseLabelMatrix(shape={self.shape}, "
                f"labeled={self.count_labeled()}, "
                f"pos={self.count_positive()}, "
                f"neg={self.count_negative()})")


class RowView:
    """
    Lazy row view for mat[Lid][Rid_arr].
    mat[Lid][Rid_arr] calls RowView(mat, Lid).__getitem__(Rid_arr).
    """
    def __init__(self, matrix: SparseLabelMatrix, Lid: int):
        self.matrix = matrix
        self.Lid = Lid

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            # mat[Lid][Rid]
            return self.matrix.get(self.Lid, int(key))
        elif isinstance(key, (list, np.ndarray)):
            # mat[Lid][Rid_arr]
            return self.matrix.get_row_labels(self.Lid, np.asarray(key))
        elif isinstance(key, slice):
            # mat[Lid][start:stop] is kept for compatibility but is unused.
            raise NotImplementedError("Slice indexing not supported on sparse matrix rows")
        else:
            raise TypeError(f"Unsupported row indexing: {type(key)}")
