import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import argparse
import tqdm
import time
from torch import optim




import os
import numpy as np
import pandas as pd
import glob
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
import warnings
from utils.augmentation import run_augmentation_single

# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_f1, evaluate_calibration
from leaderboard import display_leaderboard, load_leaderboard_data
from pdm_loader import PDMloader
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, MLP
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
DATASET_DIR = './dataset/'
RESULTS_DIR = './results/'
LOGS_DIR = './logs/'
CHECKPOINTS_DIR = './checkpoints/'
MODELS_DIR = './models/'
dataset_mapping = {
        "01": {
            "name": "Paderborn",
            "description": "Paderborn University Bearing Dataset – High-resolution vibration and motor current signals for 26 damaged and 6 healthy bearing states under various operating conditions.",
            "features": ["vibration", "current", "temperature", "speed", "torque", "radial_load"],
            "fault_types": ["normal", "inner_race", "outer_race", "cage"],
            "sampling_rate": 64000,
            "duration": 4,
            "channels": 3,
            "design_target": "Fault diagnosis",
            "property": "Multiple sensors"
        },
        "02": {
            "name": "CWRU",
            "description": "Case Western Reserve University Bearing Dataset – Vibration data collected under various fault conditions for bearing fault diagnosis.",
            "features": ["vibration"],
            "fault_types": ["normal", "inner_race", "outer_race", "ball"],
            "sampling_rate": 12000,
            "duration": 10,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "03": {
            "name": "XJTU-SY",
            "description": "Xi'an Jiaotong University Bearing Dataset – Run-to-failure vibration data for bearing fault diagnosis and remaining useful life prediction.",
            "features": ["vibration"],
            "fault_types": ["normal", "inner_race", "outer_race", "cage"],
            "sampling_rate": 25600,
            "duration": 8,
            "channels": 1,
            "design_target": "RUL prediction & Fault diagnosis",
            "property": "Vibration"
        },
        "04": {
            "name": "IMS",
            "description": "Intelligent Maintenance Systems Bearing Dataset – Long-term run-to-failure vibration data for bearing prognostics research.",
            "features": ["vibration"],
            "fault_types": ["normal", "failure"],
            "sampling_rate": 20480,
            "duration": 24,
            "channels": 1,
            "design_target": "RUL prediction",
            "property": "Vibration"
        },
        "05": {
            "name": "FEMTO",
            "description": "FEMTO-ST Institute Bearing Dataset – Accelerated degradation tests with vibration and temperature data for bearing fault diagnosis.",
            "features": ["vibration", "temperature"],
            "fault_types": ["normal", "failure"],
            "sampling_rate": 25600,
            "duration": 6,
            "channels": 2,
            "design_target": "RUL prediction",
            "property": "Multiple sensors"
        },
        "06": {
            "name": "MFPT",
            "description": "Mechanical Fault Prevention Technology Bearing Dataset – High-frequency vibration data for various bearing fault conditions.",
            "features": ["vibration"],
            "fault_types": ["normal", "inner_race", "outer_race", "ball"],
            "sampling_rate": 97600,
            "duration": 2,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "07": {
            "name": "HUST Bearing",
            "description": "Hanoi University of Science and Technology Bearing Dataset – A practical dataset for ball bearing fault diagnosis, encompassing various defect types across multiple bearing models and operating conditions.",
            "features": ["vibration"],
            "fault_types": ["inner_crack", "outer_crack", "ball_crack", "inner_outer_combination", "inner_ball_combination", "outer_ball_combination"],
            "sampling_rate": 51200,
            "duration": 10,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "09": {
            "name": "Electric Motor",
            "description": "Electric Motor Fault Dataset – Multisensor data for diagnosing faults in electric motors, including bearing, rotor, and stator faults.",
            "features": ["current", "vibration", "temperature", "speed"],
            "fault_types": ["normal", "bearing", "rotor", "stator"],
            "sampling_rate": 42000,
            "duration": 5,
            "channels": 4,
            "design_target": "Fault diagnosis",
            "property": "Multiple sensors"
        },
        "12": {
            "name": "Rotor Broken Bar",
            "description": "IEEE Rotor Broken Bar Dataset – Current measurements for detecting broken rotor bar faults in three-phase induction motors.",
            "features": ["current"],
            "fault_types": ["normal", "broken_bar"],
            "sampling_rate": 50000,
            "duration": 3,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Current"
        },
        "13": {
            "name": "WT Planetary Gearbox",
            "description": "Wind Turbine Planetary Gearbox Dataset – Vibration data for diagnosing faults in planetary gearboxes under variable operating conditions.",
            "features": ["vibration"],
            "fault_types": ["normal", "sun_gear", "planet_gear", "ring_gear"],
            "sampling_rate": 48000,
            "duration": 6,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "16": {
            "name": "CQU Gearbox",
            "description": "Chongqing University Gearbox Dataset – Multisensor data for diagnosing faults in gearboxes, including gear wear and breakage.",
            "features": ["vibration", "temperature", "speed"],
            "fault_types": ["normal", "gear_wear", "gear_break", "bearing"],
            "sampling_rate": 20000,
            "duration": 8,
            "channels": 3,
            "design_target": "Fault diagnosis",
            "property": "Multiple sensors"
        },
        "17": {
            "name": "UConn Gearbox",
            "description": "University of Connecticut Gearbox Dataset – Vibration data for diagnosing gear wear and breakage in gearboxes.",
            "features": ["vibration"],
            "fault_types": ["normal", "gear_wear", "gear_break"],
            "sampling_rate": 20000,
            "duration": 4,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "18": {
            "name": "MAFAULDA",
            "description": "Multiple Fault Dataset – Multisensor data for diagnosing various faults in rotating machinery, including bearing, rotor, stator, and gear faults.",
            "features": ["vibration", "current", "temperature", "speed"],
            "fault_types": ["normal", "bearing", "rotor", "stator", "gear"],
            "sampling_rate": 51200,
            "duration": 12,
            "channels": 4,
            "design_target": "Fault diagnosis",
            "property": "Multiple sensors"
        },
        "19": {
            "name": "Mendeley Bearing",
            "description": "Mendeley Bearing Dataset – Vibration data for diagnosing faults in rolling element bearings under varying speed conditions.",
            "features": ["vibration"],
            "fault_types": ["normal", "inner_race", "outer_race", "ball"],
            "sampling_rate": 9600,
            "duration": 3,
            "channels": 1,
            "design_target": "Fault diagnosis",
            "property": "Vibration"
        },
        "14": {
            "name": "Microsoft Azure",
            "description": "Microsoft Azure Predictive Maintenance Dataset – Telemetry and log data for predicting machine failures and maintenance needs.",
            "features": ["telemetry", "errors", "maintenance", "machines"],
            "fault_types": ["normal", "failure"],
            "sampling_rate": 1,
            "duration": 365,
            "channels": 4,
            "design_target": "RUL prediction",
            "property": "Telemetry, Logs"
        }
    }
model_dict = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'DLinear': DLinear,
    'FEDformer': FEDformer,
    'Informer': Informer,
    'LightTS': LightTS,
    'Reformer': Reformer,
    'ETSformer': ETSformer,
    'PatchTST': PatchTST,
    'Pyraformer': Pyraformer,
    'MICN': MICN,
    'Crossformer': Crossformer,
    'FiLM': FiLM,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'TiDE': TiDE,
    'FreTS': FreTS,
    'MambaSimple': MambaSimple,
    'TimeMixer': TimeMixer,
    'TSMixer': TSMixer,
    'SegRNN': SegRNN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    "SCINet": SCINet,
    'PAttn': PAttn,
    'TimeXer': TimeXer,
    'WPMixer': WPMixer,
    'MultiPatchFormer': MultiPatchFormer,
    'MLP': MLP
}


def data_provider(args, flag):
    Data = PdMDataset
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            file_list=args.file_list,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            # prefetch_factor=2
            # collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


# Ensure directories exist
for directory in [DATASET_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# class PdMDataset(Dataset):
#     """Dataset class for Predictive Maintenance data"""
#     def __init__(self, data, labels=None, feature_df=None, class_names=None, max_seq_len=None):
#         print("dataset init")
#         num_samples = len(labels)
#         num_features = len(data) // num_samples
#         # data = data.values.reshape(num_samples, num_features)
#         self.data = torch.FloatTensor(data = data.values.reshape(num_samples, num_features))
#         self.labels = torch.LongTensor(labels.values).squeeze() if labels is not None else None
#         print(self.data.shape, self.labels.shape, "dataset init done")
#         # print("dataset init done")
#         # self.feature_df = feature_df
#         # self.class_names = class_names
#         # self.max_seq_len = max_seq_len if max_seq_len is not None else data.shape[1]
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, ind):
#         batch_x = self.feature_df.loc[self.all_IDs[ind]].values
#         labels = self.labels_df.loc[self.all_IDs[ind]].values
#         if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
#             num_samples = len(self.all_IDs)
#             num_columns = self.feature_df.shape[1]
#             seq_len = int(self.feature_df.shape[0] / num_samples)
#             batch_x = batch_x.reshape((1, seq_len, num_columns))
#             batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

#             batch_x = batch_x.reshape((1 * seq_len, num_columns))

#         return self.instance_norm(torch.from_numpy(batch_x)), \
#                torch.from_numpy(labels)


class PdMDataset(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        print("loading all", root_path, file_list, flag)
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

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
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.npz')]
        if len(input_paths) == 0:
            pattern='*.csv'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        # df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
        #                                                      replace_missing_vals_with='NaN')
        print("loading single", filepath)
        loaded = np.load(filepath, allow_pickle=True)
        df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
        # df = pd.DataFrame(df_restored['features'].tolist())
        df = pd.DataFrame(df_restored['features'])
        labels = df_restored['label']

        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]
        # self.max_seq_len = int(df.shape[1])

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
    
    

class DataProvider:
    """Data provider for Predictive Maintenance tasks"""
    def __init__(self, args):
        self.args = args
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.features = args.features
        self.target = args.target
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Use the complete dataset mapping provided by the user
        self.dataset_mapping =  dataset_mapping

    
    def _load_and_preprocess_data(self, flag: str, model_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame, List[str], int]:
        """Load and preprocess data for a specific split"""
        # Construct file path based on flag
        file_name = f'{model_id}/PdM_{flag}.npz'
        file_path = os.path.join(self.root_path, file_name)
        loaded = np.load(file_path, allow_pickle=True)
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
        # df = grp.transform(self.interpolate_missing)
        return df, labels_df

        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # try:
        #     # Use PDMloader to load the data
        #     loader = PDMloader(
        #         args=self.args,
        #         root_path=self.root_path,
        #         file_list=[file_name],
        #         flag=flag
        #     )
            
        #     # Get feature names and class names
        #     dataset_info = self.dataset_mapping.get(self.args.data, {})
        #     feature_names = dataset_info.get('features', [f'feature_{i}' for i in range(loader.feature_df.shape[1])])
        #     class_names = dataset_info.get('fault_types', loader.class_names)
            
        #     # Create feature DataFrame
        #     feature_df = pd.DataFrame(columns=feature_names)
        #     print(feature_df)
        #     # Get data and labels
        #     x = loader.feature_df.values
        #     y = loader.labels_df.values if loader.labels_df is not None else None
            
        #     # Normalize features if needed
        #     if self.args.normalize:
        #         try:
        #             if flag == 'TRAIN':
        #                 x_reshaped = x.reshape(-1, x.shape[-1])
        #                 self.scaler.fit(x_reshaped)
        #             x_reshaped = x.reshape(-1, x.shape[-1])
        #             x_normalized = self.scaler.transform(x_reshaped)
        #             x = x_normalized.reshape(x.shape)
        #         except Exception as e:
        #             raise Exception(f"Error during normalization: {e}")
            
        #     # Encode labels if needed
        #     if y is not None:
        #         try:
        #             if flag == 'TRAIN':
        #                 self.label_encoder.fit(y)
        #             y = self.label_encoder.transform(y)
        #         except Exception as e:
        #             raise Exception(f"Error during label encoding: {e}")
            
        #     # Calculate max sequence length
        #     max_seq_len = loader.max_seq_len
            
        #     return x, y, feature_df, class_names, max_seq_len
            
        # except Exception as e:
        #     raise Exception(f"Error loading data from {file_path}: {e}")
    
    def get_data(self, flag: str,model_id: str) -> Tuple[PdMDataset, DataLoader]:
        """Get data and dataloader for a specific split"""
        # try:
        #     x, y = self._load_and_preprocess_data(flag, model_id)
        # except Exception as e:
        #     raise Exception(f"Error loading data for {flag}: {e}")
        # print(x, y, "loaded x and y")
        # Create dataset
        # dataset = PdMDataset(
        #     data=x,
        #     labels=y,
        #     # feature_df=feature_df,
        #     # class_names=class_names,
        #     # max_seq_len=max_seq_len
        # )
        print("root path", self.args.root_path)
        dataset= PdMDataset(
            args = self.args,
            root_path=self.args.root_path,
            file_list=self.args.file_list,
            flag=flag,
        )
        # print("dataset created")
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(flag == 'TRAIN'),
            num_workers=self.args.num_workers,
            drop_last=(flag == 'TRAIN')
        )
        
        return dataset, dataloader


def get_dataset_info(dataset_id: str) -> Dict:
    class Args:
        def __init__(self):
            self.root_path = DATASET_DIR
            self.data = None
            self.data_path = DATASET_DIR
            self.features = None
            self.target = None
            self.batch_size = 32
            self.num_workers = 0
            self.normalize = True

    provider = DataProvider(Args())
    dataset_mapping = provider.dataset_mapping
    return dataset_mapping.get(dataset_id, {})


def get_available_datasets(root_path: str = './dataset/') -> Dict:
    """Get list of available datasets"""
    class Args:
        def __init__(self):
            self.root_path = root_path
            self.data = None
            self.data_path = root_path
            self.features = None
            self.target = None
            self.batch_size = 32
            self.num_workers = 0
            self.normalize = True
    
    provider = DataProvider(Args())
    return {k: v['name'] for k, v in provider.dataset_mapping.items()}

# --------------------
# Experiment Class
# --------------------
class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.wandb = None # Initialize wandb
        if args.use_wandb:
            wandb.init(project="Your_Project_Name", name=args.model_id) # Change Project Name

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.gpu_type == 'cuda':
                device = torch.device('cuda:{}'.format(self.args.gpu))
            elif self.args.gpu_type == 'mps':
                device = torch.device('mps')
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
            logging.info('Use GPU: {}'.format(device))
        else:
            device = torch.device('cpu')
            logging.info('Use CPU')
        return device

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self, flag):
        raise NotImplementedError
        return None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()  # Default, can be overridden
        return criterion

    def train(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

# ----------------------------
# Classification Experiment
# ----------------------------
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.args = args

    def _build_model(self):
        # model input depends on data
        logging.info('Loading dataset!')
        self.args.root_path = f'{self.args.root_path}{self.args.data}/'
        print("root path", self.args.root_path)
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        
        self.train_data = train_data
        self.train_loader = train_loader

        self.vali_data = vali_data
        self.vali_loader = vali_loader

        self.test_data = test_data
        self.test_loader = test_loader

        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = max(len(train_data.class_names), len(test_data.class_names))
        # self.args.num_class = len(torch.unique(train_data.labels).tolist())
        # model inita
       
        model = model_dict[self.args.model].Model(self.args).float().to(self.device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
 
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader
    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train(self):
        train_data = self.train_data
        train_loader = self.train_loader
        vali_data = self.vali_data
        vali_loader = self.vali_loader
        test_data = self.test_data
        test_loader = self.test_loader
        print("start training")
        path = os.path.join(self.args.checkpoints, self.args.root_path.split("/")[-2], self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        print("early stopping initialized")
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        epoch_time_list = []
        print("epoch time list initialized")
        for epoch in tqdm.tqdm(range(self.args.train_epochs)):
            train_loss = []
            preds = []
            trues = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, label) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            
            epoch_time_list.append((time.time() - start_time))
            
            if len(preds) != 0:
                preds = torch.cat(preds, 0)
                trues = torch.cat(trues, 0)
                probs = torch.nn.functional.softmax(preds, dim=1)
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
                trues = trues.flatten().cpu().numpy()
                train_accuracy = cal_accuracy(predictions, trues)
            else:
                train_accuracy = 0.0
            
            train_loss = np.average(train_loss)
            val_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            logging.info(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                         f"Vali Loss: {val_loss:.4f}, Vali Acc: {val_accuracy:.4f}, "
                         f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
            if self.wandb is not None:
                self.wandb.log(
                    {
                        "Epoch": epoch + 1,
                        "Loss/Train": train_loss,
                        "Loss/Val": val_loss,
                        "Loss/Test": test_loss,
                        "Acc/Train": train_accuracy,
                        "Acc/Val": val_accuracy,
                        "Acc/Test": test_accuracy,
                        "Time Per Epoch": epoch_time_list[-1]
                    },
                    commit=True
                )
            
            early_stopping(val_accuracy, self.model, path)
            if early_stopping.early_stop:
                logging.info(f"Current epoch: {epoch + 1}")
                logging.info("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        logging.info(f"Average training time per epoch: {np.mean(np.array(epoch_time_list))}")
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in tqdm.tqdm(enumerate(vali_loader), desc="Validation"):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, label.long().squeeze().reshape(-1,))
                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu())
                trues.append(label)

        total_loss = np.average(total_loss)

        if len(preds) != 0:
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            probs = torch.nn.functional.softmax(preds, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            trues = trues.flatten().cpu().numpy()
            accuracy = cal_accuracy(predictions, trues)
        else:
            accuracy = 0.0

        return total_loss, accuracy
    
    def test(self, load_model=False):
        if load_model:
            logging.info('Loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, self.args.root_path.split("/")[-2], self.args.model, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in tqdm.tqdm(enumerate(self.test_loader), desc="Testing"):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, None, None, None)
                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        logging.info(f'Test shape: {preds.shape}, {trues.shape}')

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        f1_micro, f1_macro, f1_weighted = cal_f1(predictions, trues)
        nll, ece, brier = evaluate_calibration(torch.tensor(np.nan_to_num(probs.cpu(), nan=0.0), dtype=torch.float32), torch.tensor(trues, dtype=torch.float32))

        # result save
        folder_path = os.path.join('./results/', self.args.root_path.split("/")[-2], self.args.model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        logging.info(f'Accuracy: {accuracy}, F1 Micro: {f1_micro}, ECE: {ece}, NLL: {nll}, Brier: {brier}')
        file_name = 'result_classification.txt'
        with open(os.path.join(folder_path, file_name), 'w') as f:
            args_dict = vars(self.args)
            for key, value in args_dict.items():
                f.write(f"{key}:{value}\n")
            f.write("\n\n")
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'F1 Micro: {f1_micro}\n')
            f.write(f'F1 Macro: {f1_macro}\n')
            f.write(f'F1 Weighted: {f1_weighted}\n')
            f.write(f'ECE: {ece}\n')
            f.write(f'NLL: {nll}\n')
            f.write(f'Brier: {brier}\n')
        return
# ----------------------
# Utility Functions
# ----------------------


def get_args():
    """Helper function to define default arguments.  These can be overridden by user selections in the Streamlit app."""
    class Args:
        def __init__(self):
            self.task_name = 'classification'  # Or whatever the default task is
            self.is_training = 1
            self.model_id = 'default_model_id'
            self.model = 'LSTM'  #  default model
            self.data = '01'  # default dataset
            self.root_path = DATASET_DIR
            self.file_list = ['PdM_TRAIN.npz', 'PdM_VAL.npz', 'PdM_TEST.npz']
            self.data_path = os.path.join(DATASET_DIR, 'PdM_01')
            self.features = None
            self.target = None
            self.freq = 'h'
            self.checkpoints = CHECKPOINTS_DIR
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            self.mask_rate = 0.25
            self.anomaly_ratio = 0.25
            self.expand = 2
            self.d_conv = 4
            self.top_k = 5
            self.num_kernels = 6
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 2048
            self.moving_avg = 25
            self.factor = 1
            self.distil = True
            self.dropout = 0.1
            self.embed = 'timeF'
            self.activation = 'gelu'
            self.channel_independence = 1
            self.decomp_method = 'moving_avg'
            self.use_norm = 1
            self.down_sampling_layers = 0
            self.down_sampling_window = 1
            self.down_sampling_method = None
            self.seg_len = 96
            self.project_input_shape = 96
            self.num_workers = 0  # Streamlit doesn't handle multiprocessing well.
            self.itr = 1
            self.train_epochs = 2
            self.batch_size = 32
            self.patience = 3
            self.learning_rate = 0.001
            self.des = 'default_experiment'
            self.loss = 'CrossEntropyLoss'  # Default loss
            self.lradj = 'type1'
            self.use_amp = False
            self.use_gpu = torch.cuda.is_available()
            self.gpu = 0
            self.gpu_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.use_multi_gpu = False
            self.devices = '0'
            self.use_wandb = False
            self.p_hidden_dims = [128, 128]
            self.p_hidden_layers = 2
            self.use_dtw = False
            self.augmentation_ratio = 0
            self.seed = 2
            self.jitter = False
            self.scaling = False
            self.permutation = False
            self.randompermutation = False
            self.magwarp = False
            self.timewarp = False
            self.windowslice = False
            self.windowwarp = False
            self.rotation = False
            self.spawner = False
            self.dtwwarp = False
            self.shapedtwwarp = False
            self.wdba = False
            self.discdtw = False
            self.discsdtw = False
            self.extra_tag = ""
            self.patch_len = 16
    return Args()




def get_dataset_list() -> Dict:
    """Get list of available datasets"""
    return get_available_datasets()


def load_dataset(dataset_id: str, args: argparse.Namespace) -> DataProvider:
    """Load a dataset using the DataProvider"""
    # args = get_args() # Removed this, now passing args
    args.root_path = DATASET_DIR
    args.data = dataset_id
    args.data_path = os.path.join(DATASET_DIR, f'PdM_{dataset_id}')
    args.features = None
    args.target = None
    args.batch_size = 32
    args.num_workers = 0
    args.normalize = True
    
    try:
        provider = DataProvider(args)
        return provider
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_id}: {e}")
        return None  # Explicitly return None on error

def plot_dataset(dataset_id: str, args: argparse.Namespace):
    """Plot dataset visualizations"""
    provider = load_dataset(dataset_id, args) # Pass the args
    if provider is None:
        st.error("Failed to load dataset")
        return
    
    try:
        train_dataset, _ = provider.get_data('TRAIN', dataset_id)
    except Exception as e:
        st.error(f"Error getting training data: {e}")
        return
    
    # Time series plot
    st.subheader("Time Series Plot")
    n_features = train_dataset.data.shape[-1]
    feature_idx = st.selectbox(
        "Select Feature",
        range(n_features),
        format_func=lambda x: f"Feature {x+1}"
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=train_dataset.feature_df[0, feature_idx],
        mode='lines',
        name='Sample 1'
    ))
    fig.update_layout(
        title=f"Time Series Plot - Feature {feature_idx+1}",
        xaxis_title="Time Step",
        yaxis_title="Value"
    )
    st.plotly_chart(fig)
    
    # Feature distribution
    st.subheader("Feature Distribution")
    fig = px.histogram(
        x=train_dataset.feature_df[:, feature_idx].flatten(),
        title=f"Distribution - Feature {feature_idx+1}"
    )
    st.plotly_chart(fig)
    print(train_dataset.labels_df)
    # Label distribution if available
    if train_dataset.labels_df is not None:
        st.subheader("Label Distribution")
        fig = go.Figure(data=[
            go.Bar(
                x=train_dataset.class_names,
                y=[(train_dataset.labels_df == i).sum().item() for i in range(len(train_dataset.class_names))],
                text=[(train_dataset.labels_df == i).sum().item() for i in range(len(train_dataset.class_names))],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Label Distribution",
            xaxis_title="Class",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)

def get_model_list() -> List:
    """Get list of available models"""
    return ["LSTM", "GRU", "Transformer"]

def get_config_options() -> Dict:
    """Get model configuration options"""
    return {
        "batch_size": [16, 32, 64, 128],
        "learning_rate": [0.0001, 0.001, 0.01],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4],
        "epochs": [10, 20, 50, 100]  # Add epochs to config options
    }

def dataset_visualization():
    """Dataset exploration and visualization component"""
    st.title("📊 Dataset Visualization")

    datasets = get_available_datasets()
    if not datasets:
        st.error("No datasets found. Please check the dataset directory.")
        return

    dataset_id = st.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        format_func=lambda x: f"{x} - {datasets[x]}"
    )
    
    dataset_info = get_dataset_info(dataset_id)

    if dataset_info:
        st.subheader("📄 Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🧾 Name:** {dataset_info.get('name', 'N/A')}")
            st.markdown(f"**📦 Features:** {', '.join(dataset_info.get('features', ['N/A']))}")
            st.markdown(f"**🔍 Fault Types:** {', '.join(dataset_info.get('fault_types', ['N/A']))}")
            st.markdown(f"**🧪 Design Target:** {dataset_info.get('design_target', 'N/A')}")
            st.markdown(f"**🏷️ Property:** {dataset_info.get('property', 'N/A')}")
        with col2:
            st.markdown(f"**📈 Sampling Rate:** {dataset_info.get('sampling_rate', 'N/A')} Hz")
            st.markdown(f"**⏱️ Duration:** {dataset_info.get('duration', 'N/A')} sec")
            st.markdown("**📝 Description:**")
            st.markdown(f"{dataset_info.get('description', 'N/A')}")

    
        args = get_args()
        args.root_path = f'{args.root_path}{dataset_id}/'
        provider = DataProvider(args)
        print("loading dataset")
        train_dataset, _ = provider.get_data('TRAIN', dataset_id)
        print("loaded dataset")
        st.subheader("📐 Feature-wise Distribution and Signals")
        print(train_dataset.feature_df.shape)
        n_features = train_dataset.feature_df.shape[-1]
        feature_idx = st.selectbox(
            "Select Feature",
            range(n_features),
            format_func=lambda x: f"Feature {x+1}"
        )
        # print("feature seleccted, flattening")
        # selected_feature = train_dataset.feature_df[:, :, feature_idx].flatten()

        # # Distribution Histogram
        # st.subheader(f"📊 Distribution of Feature {feature_idx+1}")
        # fig = px.histogram(
        #     x=selected_feature,
        #     nbins=100,
        #     title=f"Distribution - Feature {feature_idx+1}"
        # )
        # st.plotly_chart(fig)
        # print("drawing distribution")

        # Infer class names from unique labels
        print(train_dataset.labels_df.iloc[:, 0].unique())
        unique_classes = train_dataset.labels_df.iloc[:, 0].unique().tolist()
        class_names = train_dataset.class_names

        # Label Distribution Pie Chart
        if train_dataset.labels_df is not None:
            st.subheader("🎯 Class Distribution")
            labels = train_dataset.labels_df.squeeze()
            class_counts = [(labels == i).sum().item() for i in unique_classes]
            fig = px.pie(
                names=class_names,
                values=class_counts,
                title="Class Distribution",
                hole=0.4
            )
            st.plotly_chart(fig)

        st.subheader("📉 Sample Time Domain Signals by Class")
        selected_class = st.selectbox(
            "Choose Class",
            options=unique_classes,
            format_func=lambda x: f"Class {x}"
        )
    # Get indices of samples belonging to the selected class
    class_indices = (train_dataset.labels_df.iloc[:, 0] == selected_class).values
    class_indices = np.where(class_indices)[0]  # Get the actual indices where condition is True
    
    # Allow user to select a specific sample from the class
    if len(class_indices) > 0:
        # Create a selection box for sample index
        sample_idx = st.selectbox(
            f"Select a sample from class {selected_class}",
            range(min(10, len(class_indices))),
            format_func=lambda i: f"Sample {i+1}"
        )
        
        # Get the actual index in the dataset
        selected_sample_idx = class_indices[sample_idx]
        
        # Display sample information
        st.write(f"Selected Sample ID: {train_dataset.all_IDs[selected_sample_idx]}")
        
        # Visualize raw time domain signal (all features)
        st.subheader("Time Domain Visualization")
        fig = go.Figure()
        
        # Get the number of features/channels in the data
        num_features = train_dataset.feature_df.shape[1]
        
        # Plot each feature/channel
        for feat in range(num_features):
            # Get the feature data for the selected sample
            feature_data = train_dataset.feature_df.loc[train_dataset.all_IDs[selected_sample_idx]].values
            fig.add_trace(go.Scatter(
                y=feature_data[:, feat],
                mode='lines',
                name=f'Feature {feat+1}'
            ))
        
        fig.update_layout(
            title=f"Time Domain Signal - Class: {selected_class}",
            xaxis_title="Time",
            yaxis_title="Amplitude"
        )
        st.plotly_chart(fig)
        
        # FFT Analysis
        st.subheader("Frequency Domain Visualization")
        import scipy.fft
        
        # Create tabs for different features
        feature_tabs = st.tabs([f"Feature {i+1}" for i in range(num_features)])
        
        # For each feature, show the FFT
        for feat_idx, tab in enumerate(feature_tabs):
            with tab:
                # Get the signal for the selected feature
                feature_data = train_dataset.feature_df.loc[train_dataset.all_IDs[selected_sample_idx]].values
                signal = feature_data[:, feat_idx]
                
                # Calculate FFT
                fft_vals = np.abs(scipy.fft.fft(signal))
                freqs = scipy.fft.fftfreq(len(fft_vals), 1 / dataset_info['sampling_rate'])
                
                # Only show positive frequencies (first half)
                pos_freq_idx = len(freqs) // 2
                
                # Create FFT plot
                fft_fig = go.Figure()
                fft_fig.add_trace(go.Scatter(
                    x=freqs[:pos_freq_idx], 
                    y=fft_vals[:pos_freq_idx],
                    mode='lines'
                ))
                
                fft_fig.update_layout(
                    title=f"FFT Spectrum - Feature {feat_idx+1}",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Amplitude"
                )
                
                # Add option to use log scale for amplitude
                use_log_scale = st.checkbox(f"Use log scale for amplitude (Feature {feat_idx+1})")
                if use_log_scale:
                    fft_fig.update_layout(yaxis_type="log")
                
                st.plotly_chart(fft_fig)
                
                # Show dominant frequencies
                st.subheader(f"Dominant Frequencies (Feature {feat_idx+1})")
                
                # Find peaks in the FFT
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(fft_vals[:pos_freq_idx], height=np.max(fft_vals[:pos_freq_idx])/10)
                
                # Sort peaks by amplitude
                peak_freqs = freqs[peaks]
                peak_amps = fft_vals[peaks]
                
                # Sort by amplitude (descending)
                sorted_idx = np.argsort(-peak_amps)
                sorted_freqs = peak_freqs[sorted_idx]
                sorted_amps = peak_amps[sorted_idx]
                
                # Display top 5 frequencies
                top_n = min(5, len(sorted_freqs))
                if top_n > 0:
                    freq_df = pd.DataFrame({
                        "Frequency (Hz)": sorted_freqs[:top_n],
                        "Amplitude": sorted_amps[:top_n]
                    })
                    st.table(freq_df)
                else:
                    st.write("No significant peaks detected")

    # else:
    #     st.warning(f"No samples found for class {selected_class}")
        # class_indices = (train_dataset.labels_df == selected_class).nonzero().squeeze()
        # cur_data = train_dataset.feature_df.reshape(len(train_dataset.all_IDs), -1)[class_indices]
        # if class_indices.dim() == 0:
        #     class_indices = class_indices.unsqueeze(0)
        # for i in range(min(3, len(class_indices))):
        #     fig = go.Figure()
        #     for feat in range(cur_data):
        #         fig.add_trace(go.Scatter(
        #             y=train_dataset.feature_df[class_indices[i], feat],
        #             mode='lines',
        #             name=f'Feature {feat+1}'
        #         ))
        #     fig.update_layout(
        #         title=f"Sample {i+1} - Class: {selected_class}",
        #         xaxis_title="Time",
        #         yaxis_title="Amplitude"
        #     )
        #     st.plotly_chart(fig)

        # # Optional: FFT
        # if st.checkbox("Show Frequency Domain (FFT) of First Sample"):
        #     import scipy.fft
        #     signal = train_dataset.data[class_indices[0], 0].numpy()
        #     fft_vals = np.abs(scipy.fft.fft(signal))
        #     freqs = np.fft.fftfreq(len(fft_vals), 1 / dataset_info['sampling_rate'])

        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_vals[:len(freqs)//2]))
        #     fig.update_layout(title="FFT Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        #     st.plotly_chart(fig)

        # print("drawing distribution")
        # # Label Distribution Pie Chart
        # if train_dataset.labels is not None:
        #     st.subheader("🎯 Class Distribution")
        #     class_counts = [(train_dataset.labels == i).sum().item() for i in range(len(train_dataset.class_names))]
        #     fig = px.pie(
        #         names=train_dataset.class_names,
        #         values=class_counts,
        #         title="Class Distribution",
        #         hole=0.4
        #     )
        #     st.plotly_chart(fig)








        # # Signal per class
        # st.subheader("📉 Sample Time Domain Signals by Class")
        # selected_class = st.selectbox(
        #     "Choose Class",
        #     options=list(range(len(train_dataset.class_names))),
        #     format_func=lambda x: train_dataset.class_names[x]
        # )

        # class_indices = (train_dataset.labels == selected_class).nonzero().squeeze()
        # if class_indices.dim() == 0:
        #     class_indices = class_indices.unsqueeze(0)
        # for i in range(min(3, len(class_indices))):
        #     fig = go.Figure()
        #     for feat in range(n_features):
        #         fig.add_trace(go.Scatter(
        #             y=train_dataset.data[class_indices[i], :, feat],
        #             mode='lines',
        #             name=f'Feature {feat+1}'
        #         ))
        #     fig.update_layout(
        #         title=f"Sample {i+1} - Class: {train_dataset.class_names[selected_class]}",
        #         xaxis_title="Time",
        #         yaxis_title="Amplitude"
        #     )
        #     st.plotly_chart(fig)

        # # Optional: FFT
        # if st.checkbox("Show Frequency Domain (FFT) of First Sample"):
        #     import scipy.fft
        #     signal = train_dataset.data[class_indices[0], :, 0].numpy()
        #     fft_vals = np.abs(scipy.fft.fft(signal))
        #     freqs = np.fft.fftfreq(len(fft_vals), 1 / dataset_info['sampling_rate'])

        #     fig = go.Figure()
        #     fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_vals[:len(freqs)//2]))
        #     fig.update_layout(title="FFT Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        #     st.plotly_chart(fig)


def visualize_training(results: Optional[Dict]):
    """Visualize training results"""
    if results is None:
        st.warning("No results to visualize.")
        return
    
    # Plot training curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=results["train_losses"],
        mode='lines',
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        y=results["val_losses"],
        mode='lines',
        name='Validation Loss'
    ))
    fig.update_layout(
        title="Training Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )
    st.plotly_chart(fig)
    
    # Display test results
    st.subheader("Test Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Loss", f"{results['test_loss']:.4f}")
    with col2:
        st.metric("Accuracy", f"{results['accuracy']:.2f}%")

def load_results(dataset_id: str, model_type: str) -> Optional[Dict]:
    """Load saved results"""
    results_path = os.path.join(RESULTS_DIR, dataset_id, model_type, 'results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading results from {results_path}: {e}")
            st.error(f"Error loading results: {e}")
            return None
    else:
        logging.info(f"Results file not found: {results_path}")
        st.info("No saved results found. Please train the model first.")
        return None
    
# def model_visualization():
#     """Model configuration and training visualization component"""
#     st.title("🧠 Model Training")
    
#     # Get available datasets
#     datasets = get_available_datasets()
#     if not datasets:
#         st.error("No datasets found. Please check the dataset directory.")
#         return
    
#     # Dataset selection
#     dataset_id = st.selectbox(
#         "Select Dataset",
#         options=list(datasets.keys()),
#         format_func=lambda x: f"{x} - {datasets[x]}"
#     )
    
#     # Model selection - Use the full model_dict defined at the top of the file
#     model_type = st.selectbox(
#         "Select Model",
#         options=list(model_dict.keys()),
#         index=0,
#         help="Select a model architecture from the available options"
#     )
    
#     # Model configuration
#     st.subheader("⚙️ Model Configuration")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         batch_size = st.number_input("Batch Size", min_value=1, value=32)
#         learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
#         epochs = st.number_input("Epochs", min_value=1, value=10)
    
#     with col2:
#         hidden_size = st.number_input("Hidden Size", min_value=16, value=64)
#         num_layers = st.number_input("Number of Layers", min_value=1, value=2)
        
#     # Create args for the experiment
#     args = get_args()
#     args.model = model_type  # Set the selected model
#     args.data = dataset_id  # Set the selected dataset
#     args.batch_size = batch_size  # Set the batch size
#     args.train_epochs = epochs  # Set the number of epochs
#     args.d_model = hidden_size  # Set hidden size in args
#     args.learning_rate = learning_rate  # Set learning rate
#     args.e_layers = num_layers  # Set the number of encoder layers
    
#     # Training button
#     if st.button("🚀 Start Training"):
#         with st.spinner(f"Training {model_type} on dataset {datasets[dataset_id]}..."):
#             try:
#                 # Create experiment
#                 experiment = Exp_Classification(args)
                
#                 # Train and get results
#                 experiment.train()
                
#                 # Test the model
#                 experiment.test()
                
#                 # Load results
#                 results_path = os.path.join('./results/', args.root_path.split("/")[-2], args.model, 'result_classification.txt')
                
#                 if os.path.exists(results_path):
#                     with open(results_path, 'r') as f:
#                         results_text = f.read()
#                     st.success("✅ Training completed successfully!")
#                     st.subheader("📊 Results")
#                     st.text(results_text)
#                 else:
#                     st.warning("Training completed but no results file found.")
#             except Exception as e:
#                 st.error(f"❌ Error during training: {str(e)}")
#                 st.error("Try adjusting model parameters or selecting a different model.")
# import streamlit as st
# import os
# import time
# import torch
# from exp.exp_classification import Exp_Classification  # Make sure this import path is correct
# # from utils_app import get_args, get_available_datasets, model_dict


# def model_visualization():
#     """Model configuration and training visualization component with stop control"""
#     st.title("🧠 Model Training & Evaluation")

#     datasets = get_available_datasets()
#     if not datasets:
#         st.error("No datasets found. Please check the dataset directory.")
#         return

#     dataset_id = st.selectbox(
#         "Select Dataset",
#         options=list(datasets.keys()),
#         format_func=lambda x: f"{x} - {datasets[x]}"
#     )
#     model_type = st.selectbox(
#         "Select Model",
#         options=list(model_dict.keys()),
#         index=0,
#         help="Choose a model architecture for training"
#     )

#     st.subheader("⚙️ Model Configuration")
#     col1, col2 = st.columns(2)

#     with col1:
#         batch_size = st.number_input("Batch Size", min_value=1, value=32)
#         learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
#         epochs = st.number_input("Epochs", min_value=1, value=10)

#     with col2:
#         hidden_size = st.number_input("Hidden Size", min_value=16, value=64)
#         num_layers = st.number_input("Number of Layers", min_value=1, value=2)

#     args = get_args()
#     args.model = model_type
#     args.data = dataset_id
#     args.batch_size = batch_size
#     args.train_epochs = epochs
#     args.d_model = hidden_size
#     args.learning_rate = learning_rate
#     args.e_layers = num_layers
#     args.model_dict = model_dict

#     # Initialize stop flag
#     if 'stop_training' not in st.session_state:
#         st.session_state.stop_training = False

#     # Stop training button
#     if st.button("🛑 Stop Training"):
#         st.session_state.stop_training = True

#     # Start training button
#     if st.button("🚀 Start Training"):
#         st.session_state.stop_training = False  # Reset stop flag at the beginning
#         with st.spinner(f"Training {model_type} on dataset {datasets[dataset_id]}..."):
#             try:
#                 experiment = Exp_Classification(args)
#                 model = experiment.model

#                 path = os.path.join('./results/', args.root_path.split("/")[-2], args.model)
#                 os.makedirs(path, exist_ok=True)
#                 criterion = experiment._select_criterion()
#                 optimizer = experiment._select_optimizer()

#                 train_loader = experiment.train_loader
#                 epoch_time_list = []

#                 for epoch in range(args.train_epochs):
#                     if st.session_state.stop_training:
#                         st.warning("Training stopped by user.")
#                         break

#                     start_time = time.time()
#                     experiment.model.train()
#                     for batch_x, label in train_loader:
#                         optimizer.zero_grad()
#                         batch_x = batch_x.float().to(experiment.device)
#                         label = label.to(experiment.device)
#                         outputs = model(batch_x, None, None, None)
#                         loss = criterion(outputs, label.long().squeeze())
#                         loss.backward()
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
#                         optimizer.step()

#                     epoch_time_list.append(time.time() - start_time)
#                     st.info(f"Epoch {epoch+1}/{args.train_epochs} completed in {epoch_time_list[-1]:.2f} seconds")

#                 if not st.session_state.stop_training:
#                     experiment.test()
#                     results_path = os.path.join(path, 'result_classification.txt')
#                     if os.path.exists(results_path):
#                         with open(results_path, 'r') as f:
#                             st.subheader("📊 Training Results")
#                             st.text(f.read())
#                     else:
#                         st.warning("Training finished, but no results found.")
#             except Exception as e:
#                 st.error(f"❌ Error du




# def get_available_datasets():
#     """Get available datasets - assuming this function exists in the original code"""
#     # This is a placeholder - should be replaced with actual implementation
#     return {"dataset1": "Dataset 1 Name", "dataset2": "Dataset 2 Name"}



# def model_visualization():
#     """Model configuration and training visualization component"""
#     st.title("🧠 Model Training & Evaluation")

#     datasets = get_available_datasets()
#     if not datasets:
#         st.error("No datasets found. Please check the dataset directory.")
#         return

#     dataset_id = st.selectbox(
#         "Select Dataset",
#         options=list(datasets.keys()),
#         format_func=lambda x: f"{x} - {datasets[x]}"
#     )
#     model_type = st.selectbox(
#         "Select Model",
#         options=list(model_dict.keys()),
#         index=0,
#         help="Choose a model architecture for training"
#     )

#     st.subheader("⚙️ Model Configuration")
#     col1, col2 = st.columns(2)

#     with col1:
#         batch_size = st.number_input("Batch Size", min_value=1, value=32)
#         learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
#         epochs = st.number_input("Epochs", min_value=1, value=10)

#     with col2:
#         hidden_size = st.number_input("Hidden Size", min_value=16, value=64)
#         num_layers = st.number_input("Number of Layers", min_value=1, value=2)

#     args = get_args()
#     args.model = model_type
#     args.data = dataset_id
#     args.batch_size = batch_size
#     args.train_epochs = epochs
#     args.d_model = hidden_size
#     args.learning_rate = learning_rate
#     args.e_layers = num_layers
#     args.model_dict = model_dict


#     # Add a placeholder for real-time training visualization
#     training_status = st.empty()
#     progress_bar = st.empty()
#     live_metrics = st.empty()
#     live_chart = st.empty()
    
#     # Add a session state to track if training is in progress
#     if 'training_in_progress' not in st.session_state:
#         st.session_state.training_in_progress = False
#         st.session_state.stop_training = False
#         st.session_state.current_results = {
#             "train_losses": [],
#             "val_losses": [],
#             "test_loss": 0.0,
#             "accuracy": 0.0,
#             "current_epoch": 0
#         }

#     # Create columns for start and stop buttons
#     col1, col2 = st.columns(2)
    
#     with col1:
#         start_button = st.button("🚀 Start Training")
    
#     with col2:
#         stop_button = st.button("🛑 Stop Training", disabled=not st.session_state.training_in_progress)
#         if stop_button:
#             st.session_state.stop_training = True
#             st.warning("Stopping training after current epoch...")

#     if start_button and not st.session_state.training_in_progress:
#         # Reset stop flag
#         st.session_state.stop_training = False
#         st.session_state.training_in_progress = True
#         st.session_state.current_results = {
#             "train_losses": [],
#             "val_losses": [],
#             "test_loss": 0.0,
#             "accuracy": 0.0,
#             "current_epoch": 0
#         }
#         # Display container for logs
#         log_output = st.empty()

#         # Set up logging to Streamlit UI
#         streamlit_handler = StreamlitLogger(log_output)
#         streamlit_handler.setLevel(logging.INFO)
#         logging.getLogger().addHandler(streamlit_handler)

#         # Show training in progress
#         training_status.info("Training in progress...")
    
#         # Create the experiment
#         experiment = Exp_Classification(args)
#         print("experiment created, training...")
#         experiment.train()
#         print("training completed, testing...")
#         experiment.test()

#         results_path = os.path.join(
#             './results/',
#             args.root_path.split("/")[-2],
#             args.model,
#             'result_classification.txt'
#         )

        
#         # Custom training loop with progress updates
#         for epoch in range(1, epochs + 1):
#             if st.session_state.stop_training:
#                 training_status.warning("Training stopped by user.")
#                 break
            
#             # Update progress bar
#             progress = epoch / epochs
#             progress_bar.progress(progress)
            
#             # Simulate epoch training
#             train_loss, val_loss = experiment.train(epoch)
            
#             # Update session state with current results
#             st.session_state.current_results["train_losses"].append(train_loss)
#             st.session_state.current_results["val_losses"].append(val_loss)
#             st.session_state.current_results["current_epoch"] = epoch
            
#             # Update live metrics
#             col1, col2 = live_metrics.columns(2)
#             with col1:
#                 st.metric("Current Epoch", f"{epoch}/{epochs}")
#                 st.metric("Training Loss", f"{train_loss:.4f}")
#             with col2:
#                 st.metric("Validation Loss", f"{val_loss:.4f}")
            
#             # Update live chart
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 y=st.session_state.current_results["train_losses"],
#                 mode='lines',
#                 name='Training Loss'
#             ))
#             fig.add_trace(go.Scatter(
#                 y=st.session_state.current_results["val_losses"],
#                 mode='lines',
#                 name='Validation Loss'
#             ))
#             fig.update_layout(
#                 title="Training Progress",
#                 xaxis_title="Epoch",
#                 yaxis_title="Loss",
#                 height=300
#             )
#             live_chart.plotly_chart(fig)
            
#             # Add a small delay to make the UI updates visible
#             time.sleep(0.1)
        
#         # Run test if training wasn't stopped
#         if not st.session_state.stop_training:
#             test_loss, accuracy = experiment.test()
#             st.session_state.current_results["test_loss"] = test_loss
#             st.session_state.current_results["accuracy"] = accuracy
            
#             training_status.success("✅ Training completed successfully!")
        
#         # Display final results
#         visualize_training(st.session_state.current_results)
        
#         # Check for results file
#         results_path = os.path.join(
#             './results/',
#             args.root_path.split("/")[-2],
#             args.model,
#             'result_classification.txt'
#         )
        
#         if os.path.exists(results_path):
#             with open(results_path, 'r') as f:
#                 results_text = f.read()
#             st.subheader("📊 Results")
#             st.text(results_text)
            
#             # Parse and visualize metrics
#             metrics = {}
#             for line in results_text.split('\n'):
#                 if ':' in line:
#                     key, val = line.split(':', 1)
#                     try:
#                         metrics[key.strip()] = float(val.strip())
#                     except:
#                         pass
            
#             if metrics:
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
#                     st.metric("F1 (Micro)", f"{metrics.get('f1_micro', 0):.2f}")
#                 with col2:
#                     st.metric("F1 (Macro)", f"{metrics.get('f1_macro', 0):.2f}")
#                     st.metric("F1 (Weighted)", f"{metrics.get('f1_weighted', 0):.2f}")
#                 with col3:
#                     st.metric("ECE", f"{metrics.get('ece', 0):.4f}")
#                     st.metric("NLL", f"{metrics.get('nll', 0):.4f}")
#                     st.metric("Brier", f"{metrics.get('brier', 0):.4f}")
#         else:
#             st.warning("Training completed, but no result file found.")

        
#         # Reset training state
#         st.session_state.training_in_progress = False

def model_visualization():
    """Model configuration and training visualization component"""
    st.title("🧠 Model Training & Evaluation")

    datasets = get_available_datasets()
    if not datasets:
        st.error("No datasets found. Please check the dataset directory.")
        return

    dataset_id = st.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        format_func=lambda x: f"{x} - {datasets[x]}"
    )
    model_type = st.selectbox(
        "Select Model",
        options=list(model_dict.keys()),
        index=0,
        help="Choose a model architecture for training"
    )

    st.subheader("⚙️ Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, value=32)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
        epochs = st.number_input("Epochs", min_value=1, value=10)

    with col2:
        hidden_size = st.number_input("Hidden Size", min_value=16, value=64)
        num_layers = st.number_input("Number of Layers", min_value=1, value=2)

    args = get_args()
    args.model = model_type
    args.data = dataset_id
    args.batch_size = batch_size
    args.train_epochs = epochs
    args.d_model = hidden_size
    args.learning_rate = learning_rate
    args.e_layers = num_layers
    args.model_dict = model_dict

    training_status = st.empty()
    log_output = st.empty()

    class StreamlitLogger(logging.Handler):
        def __init__(self, container):
            super().__init__()
            self.container = container
            self.log_text = ""

        def emit(self, record):
            msg = self.format(record)
            self.log_text += msg + "\n"
            self.container.code(self.log_text)

    logger_handler = StreamlitLogger(log_output)
    logger_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(logger_handler)

    if st.button("🚀 Start Training"):
        training_status.info("Training in progress...")
        experiment = Exp_Classification(args)

        model = experiment.train()
        experiment.test(load_model=True)

        result_file_path = os.path.join(
            './results/',
            args.root_path.split("/")[-2],
            args.model,
            'result_classification.txt'
        )

        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as f:
                results_text = f.read()
            st.subheader("📊 Detailed Evaluation Results")
            # st.code(results_text)

            metrics = {}
            for line in results_text.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    try:
                        metrics[key.strip().lower()] = float(val.strip())
                    except:
                        continue

            # st.markdown("### 🔍 Summary of Key Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Accuracy", f"{metrics.get('accuracy', 0):.2f}")
                st.metric("📉 ECE", f"{metrics.get('ece', 0):.4f}")
                st.metric("⚠️ NLL", f"{metrics.get('nll', 0):.4f}")
            with col2:
                st.metric("📊 F1 Micro", f"{metrics.get('f1 micro', 0):.2f}")
                st.metric("🧮 F1 Macro", f"{metrics.get('f1 macro', 0):.2f}")
                st.metric("📌 F1 Weighted", f"{metrics.get('f1 weighted', 0):.2f}")
            with col3:
                st.metric("💧 Brier Score", f"{metrics.get('brier', 0):.4f}")
                st.metric("📦 Parameters", f"{metrics.get('parameters', 0):,.0f}")
                st.metric("⏱️ Time Cost", f"{metrics.get('time cost', 0):.2f} sec")
        else:
            st.warning("Training completed, but result file not found.")

        training_status.success("✅ Training completed successfully!")



# def model_visualization():
#     """Model configuration and training visualization component"""
#     st.title("🧠 Model Training & Evaluation")

#     datasets = get_available_datasets()
#     if not datasets:
#         st.error("No datasets found. Please check the dataset directory.")
#         return

#     dataset_id = st.selectbox(
#         "Select Dataset",
#         options=list(datasets.keys()),
#         format_func=lambda x: f"{x} - {datasets[x]}"
#     )
#     model_type = st.selectbox(
#         "Select Model",
#         options=list(model_dict.keys()),
#         index=0,
#         help="Choose a model architecture for training"
#     )

#     st.subheader("⚙️ Model Configuration")
#     col1, col2 = st.columns(2)

#     with col1:
#         batch_size = st.number_input("Batch Size", min_value=1, value=32)
#         learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
#         epochs = st.number_input("Epochs", min_value=1, value=10)

#     with col2:
#         hidden_size = st.number_input("Hidden Size", min_value=16, value=64)
#         num_layers = st.number_input("Number of Layers", min_value=1, value=2)

#     args = get_args()
#     args.model = model_type
#     args.data = dataset_id
#     args.batch_size = batch_size
#     args.train_epochs = epochs
#     args.d_model = hidden_size
#     args.learning_rate = learning_rate
#     args.e_layers = num_layers
#     args.model_dict = model_dict

#     if st.button("🚀 Start Training"):
#         with st.spinner(f"Training {model_type} on dataset {datasets[dataset_id]}..."):
#             experiment = Exp_Classification(args)
#             experiment.train()
#             experiment.test()

#             results_path = os.path.join(
#                 './results/',
#                 args.root_path.split("/")[-2],
#                 args.model,
#                 'result_classification.txt'
#             )

#             if os.path.exists(results_path):
#                 with open(results_path, 'r') as f:
#                     results_text = f.read()
#                 st.success("✅ Training completed successfully!")
#                 st.subheader("📊 Results")
#                 st.text(results_text)

#                 # Optional: parse and visualize metrics
#                 metrics = {}
#                 for line in results_text.split('\n'):
#                     if ':' in line:
#                         key, val = line.split(':', 1)
#                         try:
#                             metrics[key.strip()] = float(val.strip())
#                         except:
#                             pass

#                 if metrics:
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
#                         st.metric("F1 (Micro)", f"{metrics.get('f1_micro', 0):.2f}")
#                     with col2:
#                         st.metric("F1 (Macro)", f"{metrics.get('f1_macro', 0):.2f}")
#                         st.metric("F1 (Weighted)", f"{metrics.get('f1_weighted', 0):.2f}")
#                     with col3:
#                         st.metric("ECE", f"{metrics.get('ece', 0):.4f}")
#                         st.metric("NLL", f"{metrics.get('nll', 0):.4f}")
#                         st.metric("Brier", f"{metrics.get('brier', 0):.4f}")
#             else:
#                 st.warning("Training completed, but no result file found.")

#             # except Exception as e:
#             #     st.error(f"❌ Error during training: {str(e)}")
#             # st.error("Please adjust hyperparameters or verify dataset/model compatibility.")

# def results_visualization():
#     """Results analysis and visualization component"""
#     st.title("📈 Results Analysis")
    
#     # Get available datasets
#     datasets = get_available_datasets()
#     if not datasets:
#         st.error("No datasets found. Please check the dataset directory.")
#         return
    
#     # Dataset selection
#     dataset_id = st.selectbox(
#         "Select Dataset",
#         options=list(datasets.keys()),
#         format_func=lambda x: f"{x} - {datasets[x]}"
#     )
    
#     # Model selection
#     model_type = st.selectbox(
#         "Select Model",
#         ["LSTM", "GRU", "Transformer"]  # Add more models as you implement them
#     )
    
#     # Load and display results
#     results = load_results(dataset_id, model_type)
#     if results:
#         st.subheader(f"Results for {model_type} on {datasets[dataset_id]}")
        
#         # Display the results
#         st.write(f"Test Loss: {results['test_loss']:.4f}")
#         st.write(f"Accuracy: {results['accuracy']:.2f}%")
        
#         #  Show the training curves
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             y=results["train_losses"],
#             mode='lines',
#             name='Training Loss'
#         ))
#         fig.add_trace(go.Scatter(
#             y=results["val_losses"],
#             mode='lines',
#             name='Validation Loss'
#         ))
#         fig.update_layout(
#             title=f"Training Curves for {model_type} on {datasets[dataset_id]}",
#             xaxis_title="Epoch",
#             yaxis_title="Loss"
#         )
#         st.plotly_chart(fig)
        
#     else:
#         st.info("No results found for the selected dataset and model. Please train the model first.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
def results_visualization():
    st.title("📈 Results Analysis and Leaderboard")

    # Load all result files in a directory
    result_files = glob.glob('./results_tmp/*.xlsx')  # Adjust path if needed
    if not result_files:
        st.error("No results files found in './results_tmp/'.")
        return

    all_dfs = []
    for path in result_files:
        raw_code = os.path.basename(path).split("_")[0]
        dataset_code = raw_code.zfill(2)
        dataset_name = dataset_mapping.get(dataset_code, {}).get("name", f"Unknown ({dataset_code})")

        df = pd.read_excel(path)
        df['dataset_code'] = dataset_code
        df['dataset_name'] = dataset_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.columns = [col.strip() for col in combined_df.columns]
    combined_df['model'] = combined_df['model'].astype(str)

    metric_cols = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'ece', 'nll', 'brier', 'time cost', 'parameters']
    for col in metric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    st.sidebar.header("🔧 Configuration")

    available_datasets = sorted(combined_df['dataset_name'].unique())
    selected_datasets = st.sidebar.multiselect(
        "Select Datasets",
        options=available_datasets,
        default=available_datasets[:3] if len(available_datasets) > 3 else available_datasets
    )

    all_models = sorted(combined_df['model'].unique())
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=all_models,
        default=all_models[:5] if len(all_models) > 5 else all_models
    )

    filtered_df = combined_df[
        (combined_df['dataset_name'].isin(selected_datasets)) &
        (combined_df['model'].isin(selected_models))
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    tab1, tab2 = st.tabs(["📊 All Metrics View", "🏆 Leaderboard"])

    with tab1:
        st.subheader("📊 Performance Metrics Across Selected Models")
        for metric in metric_cols:
            st.subheader(f"{metric.replace('_', ' ').title()}")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=filtered_df,
                x='model',
                y=metric,
                hue='dataset_name',
                ax=ax
            )
            ax.set_title(f"{metric.replace('_', ' ').title()} by Model and Dataset")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha='right')
            ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

            pivot_table = filtered_df.pivot_table(
                values=metric,
                index='model',
                columns='dataset_name',
                aggfunc='mean'
            ).round(4)
            st.dataframe(pivot_table)
            st.markdown("---")

    with tab2:
        st.subheader("🏆 Overall Model Leaderboard")
        leaderboard_dfs = {}
        for metric in metric_cols:
            leaderboard_df = (
                filtered_df.groupby("model")[metric]
                .agg(['mean', 'std', 'count'])
                .round(4)
                .sort_values('mean', ascending=False)
                .reset_index()
            )
            leaderboard_df.columns = ['Model', f'{metric.replace("_", " ").title()} (Avg)', 
                                    f'{metric.replace("_", " ").title()} (Std)', 'Number of Datasets']
            leaderboard_dfs[metric] = leaderboard_df

        for metric in metric_cols:
            st.subheader(f"🏆 {metric.replace('_', ' ').title()} Leaderboard")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(
                data=leaderboard_dfs[metric],
                x=f'{metric.replace("_", " ").title()} (Avg)',
                y='Model',
                palette="viridis",
                ax=ax
            )
            ax.set_title(f"Model Leaderboard by {metric.replace('_', ' ').title()}")
            ax.set_xlabel(f"Average {metric.replace('_', ' ').title()}")
            ax.set_ylabel("Model")
            st.pyplot(fig)

            st.dataframe(leaderboard_dfs[metric])
            st.markdown("---")

    with st.expander("📋 Full Results Table"):
        st.dataframe(filtered_df)
# def results_visualization():
#     st.title("📈 Results Analysis and Leaderboard")

#     # Load all result files in a directory
#     result_files = glob.glob('./results_tmp/*.xlsx')  # Adjust path if needed
#     if not result_files:
#         st.error("No results files found in './results_tmp/'.")
#         return
    
#     all_dfs = []
#     for path in result_files:
#         raw_code = os.path.basename(path).split("_")[0]
#         dataset_code = raw_code.zfill(2)  # Ensures "1" becomes "01", etc.
#         dataset_name = dataset_mapping.get(dataset_code, {}).get("name", f"Unknown ({dataset_code})")

#         df = pd.read_excel(path)
#         df['dataset_code'] = dataset_code
#         df['dataset_name'] = dataset_name
#         all_dfs.append(df)

#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     combined_df.columns = [col.strip() for col in combined_df.columns]
#     combined_df['model'] = combined_df['model'].astype(str)

#     # Cast metrics
#     metric_cols = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'ece', 'nll', 'brier', 'time cost', 'parameters']
#     for col in metric_cols:
#         combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

#     # Sidebar selection
#     st.sidebar.header("🔧 Configuration")
    
#     # Dataset selection
#     available_datasets = sorted(combined_df['dataset_name'].unique())
#     selected_datasets = st.sidebar.multiselect(
#         "Select Datasets",
#         options=available_datasets,
#         default=available_datasets[:3] if len(available_datasets) > 3 else available_datasets
#     )

#     # Filter data based on selection
#     filtered_df = combined_df[combined_df['dataset_name'].isin(selected_datasets)]

#     if filtered_df.empty:
#         st.warning("No data available for the selected filters.")
#         return

#     # Create tabs for different views
#     tab1, tab2 = st.tabs(["📊 All Metrics View", "🏆 Leaderboard"])

#     with tab1:
#         st.subheader("📊 Performance Metrics Across All Models")
        
#         # Create a grid of plots for each metric
#         for metric in metric_cols:
#             st.subheader(f"{metric.replace('_', ' ').title()}")
            
#             # Create comparison plot
#             fig, ax = plt.subplots(figsize=(12, 6))
#             sns.barplot(
#                 data=filtered_df,
#                 x='model',
#                 y=metric,
#                 hue='dataset_name',
#                 ax=ax
#             )
#             ax.set_title(f"{metric.replace('_', ' ').title()} by Model and Dataset")
#             ax.set_xlabel("Model")
#             ax.set_ylabel(metric.replace('_', ' ').title())
#             plt.xticks(rotation=45, ha='right')
#             ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
#             st.pyplot(fig)

#             # Show detailed comparison table
#             pivot_table = filtered_df.pivot_table(
#                 values=metric,
#                 index='model',
#                 columns='dataset_name',
#                 aggfunc='mean'
#             ).round(4)
#             st.dataframe(pivot_table)
#             st.markdown("---")

#     with tab2:
#         st.subheader("🏆 Overall Model Leaderboard")
        
#         # Calculate average performance across selected datasets for all metrics
#         leaderboard_dfs = {}
#         for metric in metric_cols:
#             leaderboard_df = (
#                 filtered_df.groupby("model")[metric]
#                 .agg(['mean', 'std', 'count'])
#                 .round(4)
#                 .sort_values('mean', ascending=False)
#                 .reset_index()
#             )
#             leaderboard_df.columns = ['Model', f'{metric.replace("_", " ").title()} (Avg)', 
#                                     f'{metric.replace("_", " ").title()} (Std)', 'Number of Datasets']
#             leaderboard_dfs[metric] = leaderboard_df

#         # Create a grid of leaderboard plots
#         for metric in metric_cols:
#             st.subheader(f"🏆 {metric.replace('_', ' ').title()} Leaderboard")
            
#             # Create leaderboard plot
#             fig, ax = plt.subplots(figsize=(10, 8))
#             sns.barplot(
#                 data=leaderboard_dfs[metric],
#                 x=f'{metric.replace("_", " ").title()} (Avg)',
#                 y='Model',
#                 palette="viridis",
#                 ax=ax
#             )
#             ax.set_title(f"Model Leaderboard by {metric.replace('_', ' ').title()}")
#             ax.set_xlabel(f"Average {metric.replace('_', ' ').title()}")
#             ax.set_ylabel("Model")
#             st.pyplot(fig)

#             # Show leaderboard table
#             st.dataframe(leaderboard_dfs[metric])
#             st.markdown("---")

#     # Show full results table in expander
#     with st.expander("📋 Full Results Table"):
#         st.dataframe(filtered_df)



# ----------------------
# Main Application
# ----------------------
def main():
    """Main application entry point"""
    # Initialize session state first
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.loading_state = None
        st.session_state.dataset_list = None
        st.session_state.selected_dataset = None
        st.session_state.selected_model = None
        st.session_state.training_started = False
        st.session_state.training_completed = False
        st.session_state.training_metrics = []
        st.session_state.test_results = None
        st.session_state.leaderboard_data = load_leaderboard_data()
        st.session_state.dataset_dict = None
        st.session_state.dataset_info = None
        st.session_state.dataset_data = None
        st.session_state.args = get_args() # Store the args
        
    
    # Initialize app if not already done
    if not st.session_state.initialized:
        show_loading_state("Initializing application...")
        try:
            data_loader = DataProvider(st.session_state.args) # Pass the args here
            st.session_state.dataset_dict = data_loader.dataset_mapping # Changed to .dataset_mapping
            st.session_state.initialized = True
            st.success(f"✅ Loaded {len(st.session_state.dataset_dict)} datasets")
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.stop()
    
    # Sidebar
    st.sidebar.title("PDM Benchmark")
    
    # Show current loading state if any
    if st.session_state.loading_state:
        st.sidebar.info(st.session_state.loading_state)
    
    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["Dataset Visualization", "Model Training", "Results Analysis"]
    )
    
    # Initialize dataset list if not already done
    if st.session_state.dataset_list is None:
        with st.spinner("Loading datasets..."):
            st.session_state.dataset_list = get_available_datasets()
    
    # Display selected page
    if page == "Dataset Visualization":
        dataset_visualization() # Pass the args
    elif page == "Model Training":
        model_visualization() # Pass the args
    else:  # Results Analysis
        results_visualization()

    # Footer
    st.markdown("---")
    st.markdown("### 📝 About")
    st.markdown("""
    PDMBench application allows you to:
    - Explore various PDM datasets
    - Configure and train state-of-the-art PDM models
    - Compare performance of different models on the leaderboard
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("📚 PDM Benchmark Tool v1.0")

def show_loading_state(message):
    """Show loading state with spinner"""
    with st.spinner(message):
        st.session_state.loading_state = message

if __name__ == "__main__":
    main()
