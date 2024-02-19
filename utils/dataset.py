import torch

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseDataset(ABC):
    """Abstract base dataset class.

    Requires subclasses to override the following:
        load, which should set all attributes below that are None
        which will be needed by the dataset (e.g. fixed_split_indices
        need only be set by a dataset that has a fully specified
        train, val, and test set for comparability to prior approaches).
    """
    def __init__(
            self, fixed_test_set_index):
        """
        Args:
            fixed_test_set_index: int, if specified, the dataset has a
                fixed test set starting at this index. Needed for
                comparability to other methods.
        """
        self.fixed_test_set_index = fixed_test_set_index
        self.c = None
        self.data_table = None
        self.missing_matrix = None
        self.N = None
        self.D = None
        self.cat_features = None
        self.num_features = None
        self.cat_target_cols = None
        self.num_target_cols = None
        self.auroc_setting = None
        self.is_data_loaded = False
        self.tmp_file_or_dir_names = []  # Deleted if c.clear_tmp_files=True

        # fixed_split_indices: Dict[str, np.array], a fully specified
        #   mapping from the dataset mode key (train, val, or test)
        #   to a np.array containing the indices for the respective
        #   mode.
        self.fixed_split_indices = None

    def get_data_dict(self, force_disable_auroc=None):
        if not self.is_data_loaded:
            self.load()

        self.auroc_setting = self.use_auroc(force_disable_auroc)

        # # # For some datasets, we should immediately delete temporary files
        # # # e.g. Higgs: zipped and unzipped file = 16 GB, CV split is 3 GB
        if self.c.data_clear_tmp_files:
            print('\nClearing tmp files.')
            path = Path(self.c.data_path) / self.c.data_set
            for file_or_dir in self.tmp_file_or_dir_names:
                file_dir_path = path / file_or_dir

                # Could be both file and a path!
                if os.path.isfile(file_dir_path):
                    os.remove(file_dir_path)
                    print(f'Removed file {file_or_dir}.')
                if os.path.isdir(file_dir_path):
                    try:
                        shutil.rmtree(file_dir_path)
                        print(f'Removed dir {file_or_dir}.')
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))

        return self.__dict__

    @abstractmethod
    def load(self):
        pass

    def use_auroc(self, force_disable=None):
        """
        Disable AUROC metric:
            (i)  if we do not have a single categorical target column,
            (ii) if the single categorical target column is multiclass.
        """
        if not self.is_data_loaded:
            self.load()

        disable = 'Disabling AUROC metric.'

        if force_disable:
            print(disable)
            return False

        if not self.c.metrics_auroc:
            print(disable)
            print("As per config argument 'metrics_auroc'.")
            return False

        num_target_cols, cat_target_cols = (
            self.num_target_cols, self.cat_target_cols)
        n_cat_target_cols = len(cat_target_cols)

        if n_cat_target_cols != 1:
            print(disable)
            print(
                f'\tBecause dataset has {n_cat_target_cols} =/= 1 '
                f'categorical target columns.')
            if n_cat_target_cols > 1:
                print(
                    '\tNote that we have not decided how we want to handle '
                    'AUROC among multiple categorical target columns.')
            return False
        elif num_target_cols:
            print(disable)
            print(
                '\tBecause dataset has a nonzero count of '
                'numerical target columns.')
            return False
        else:
            auroc_col = cat_target_cols[0]
            if n_classes := len(np.unique(self.data_table[:, auroc_col])) > 2:
                print(disable)
                print(f'\tBecause AUROC does not (in the current implem.) '
                      f'support multiclass ({n_classes}) classification.')
                return False

        return True

    @staticmethod
    def get_num_cat_auto(data, cutoff):
        """Interpret all columns with < "cutoff" values as categorical."""
        D = data.shape[1]
        cols = np.arange(0, D)
        unique_vals = np.array([np.unique(data[:, col]).size for col in cols])

        num_feats = cols[unique_vals > cutoff]
        cat_feats = cols[unique_vals <= cutoff]

        assert np.intersect1d(cat_feats, num_feats).size == 0
        assert np.union1d(cat_feats, num_feats).size == D

        # we dump to json later, it will crie if not python dtypes
        num_feats = [int(i) for i in num_feats]
        cat_feats = [int(i) for i in cat_feats]

        return num_feats, cat_feats

    @staticmethod
    def impute_missing_entries(cat_features, data_table, missing_matrix):
        """
        Fill categorical missing entries with ?
        and numerical entries with the mean of the column.
        """
        for col in range(data_table.shape[1]):
            # Get missing value locations
            curr_col = data_table[:, col]

            if curr_col.dtype == np.object_:
                col_missing = np.array(
                    [True if str(n) == "nan" else False for n in curr_col])
            else:
                col_missing = np.isnan(data_table[:, col])

            # There are missing values
            if col_missing.sum() > 0:
                # Set in missing matrix (used to avoid using data augmentation
                # or predicting on those values
                missing_matrix[:, col] = col_missing

            if col in cat_features:
                missing_impute_val = '?'
            else:
                missing_impute_val = np.mean(
                    data_table[~col_missing, col])

            data_table[:, col] = np.array([
                missing_impute_val if col_missing[i] else data_table[i, col]
                for i in range(data_table.shape[0])])

        n_missing_values = missing_matrix.sum()
        print(f'Detected {n_missing_values} missing values in dataset.')

        return data_table, missing_matrix

    def make_missing(self, p):
        N = self.N
        D = self.D

        # drawn random indices (excluding the target columns)
        target_cols = self.num_target_cols + self.cat_target_cols
        D_miss = D - len(target_cols)

        missing = np.zeros((N * D_miss), dtype=np.bool_)

        # draw random indices at which to set True do
        idxs = np.random.choice(
            a=range(0, N * D_miss), size=int(p * N * D_miss), replace=False)

        # set missing to true at these indices
        missing[idxs] = True

        assert missing.sum() == int(p * N * D_miss)

        # reshape to original shape
        missing = missing.reshape(N, D_miss)

        # add back target columns
        missing_complete = missing

        for col in target_cols:
            missing_complete = np.concatenate(
                [missing_complete[:, :col],
                 np.zeros((N, 1), dtype=np.bool_),
                 missing_complete[:, col:]],
                axis=1
            )

        if len(target_cols) > 1:
            raise NotImplementedError(
                'Missing matrix generation should work for multiple '
                'target cols as well, but this has not been tested. '
                'Please test first.')

        return missing_complete


class CIFAR10Dataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Classification dataset.

        Target in last column.
        60 000 rows.
        3072 attributes.
        1 class (10 class labels)

        Author: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
        Source: [University of Toronto]
        (https://www.cs.toronto.edu/~kriz/cifar.html) - 2009
        Alex Krizhevsky (2009) Learning Multiple Layers of Features from
            Tiny Images, Tech Report.

        CIFAR-10 is a labeled subset of the [80 million tiny images dataset]
            (http://groups.csail.mit.edu/vision/TinyImages/).

        It (originally) consists 32x32 color images representing
            10 classes of objects:
                0. airplane
                1. automobile
                2. bird
                3. cat
                4. deer
                5. dog
                6. frog
                7. horse
                8. ship
                9. truck

        CIFAR-10 contains 6000 images per class.
        Similar to the original train-test split, which randomly divided
            these classes into 5000 train and 1000 test images per class,
             we do 5-fold class-balanced cross-validation by default.

        The classes are completely mutually exclusive.
        There is no overlap between automobiles and trucks.
        "Automobile" includes sedans, SUVs, things of that sort.
        "Truck" includes only big trucks. Neither includes pickup trucks.

        ### Attribute description
        Each instance represents a 32x32 colour image as a 3072-value array.
        The first 1024 entries contain the red channel values, the next
        1024 the green, and the final 1024 the blue. The image is stored
        in row-major order, so that the first 32 entries of the array are
        the red channel values of the first row of the image.

        The labels are encoded as integers in the range 0-9,
            corresponding to the numbered classes listed above.
        """
        self.N = 60000
        self.D = 3073
        self.cat_features = [self.D - 1]
        self.num_features = list(range(0, self.D - 1))

        # Target col is the last feature
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = torch.zeros((self.N, self.D), dtype=torch.bool)
        self.is_data_loaded = True

        self.input_feature_dims = [1] * 3072
        self.input_feature_dims += [10]
