from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection._split import indexable,_num_samples,_build_repr,check_random_state
import torch

class KFold_group(KFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None,):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None, att_mask=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X=X, y=y, groups=groups, att_mask=att_mask):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index
    
    def _iter_test_masks(self, X=None, y=None, groups=None, att_mask=None):
        for test_index in self._iter_test_indices(X, y, groups, att_mask):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask
    
    def _iter_test_indices(self, X, y=None, groups=None, att_mask=None):
        "att_mask:tensor([25824,128]) attention_mask"
        n_samples = _num_samples(X) # 25824
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits 
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1 # [5165,5165,5165,5165,5164]
        length = []
        for line_index in range(len(att_mask)):
            length.append(torch.sum(att_mask[line_index]))
        current = 0
        for i,fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            if i == n_splits - 1:
                break
            back = 0
            if length[stop] == length[stop+1]:
                flag = False
                while(flag == False):
                    back += 1
                    flag = True if length[stop+back] != length[stop+back+1] else False
                fold_sizes[i] += back
                fold_sizes[i+1] -= back
            current = stop + back + 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop] 
            current = stop

