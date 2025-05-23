import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import re

class PDMloader(Dataset):
    """
    Dataset class for Predictive Maintenance datasets
    """
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        self.feature_df = self.normalize(self.feature_df)
        print(f"Loaded {len(self.all_IDs)} samples")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from npz files contained in `root_path`
        """
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.npz')]
        if len(input_paths) == 0:
            raise Exception("No .npz files found")

        all_df, labels_df = self.load_single(input_paths[0])
        return all_df, labels_df

    def load_single(self, filepath):
        """Load a single npz file"""
        loaded = np.load(filepath, allow_pickle=True)
        df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
        df = pd.DataFrame(df_restored['features'])
        labels = df_restored['label']

        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)

        lengths = df.applymap(lambda x: len(x)).values
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row has varying length across dimensions
            df = df.applymap(self.subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # Create a (seq_len, feat_dim) dataframe for each sample
        df = pd.concat((
            pd.DataFrame({col: df.loc[row, col] for col in df.columns})
            .reset_index(drop=True)
            .set_index(pd.Series(lengths[row, 0] * [row]))
            for row in range(df.shape[0])
        ), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(self.interpolate_missing)

        return df, labels_df

    def normalize(self, df):
        """Normalize the features"""
        return (df - df.mean()) / (df.std() + 1e-8)

    def interpolate_missing(self, df):
        """Interpolate missing values"""
        return df.interpolate(method='linear', limit_direction='both')

    def subsample(self, x):
        """Subsample sequence to fixed length"""
        if len(x) > self.args.seq_len:
            return x[:self.args.seq_len]
        return x

    def instance_norm(self, case):
        """Apply instance normalization"""
        mean = case.mean(0, keepdim=True)
        case = case - mean
        stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
        case /= stdev
        return case

    def __getitem__(self, ind):
        """Get a single sample"""
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values

        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            # Apply augmentation if needed
            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        """Get total number of samples"""
        return len(self.all_IDs) 