import torch
import numpy as np
import pandas as pd
import os
import pickle

from sklearn import datasets
from pathlib import Path
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset, DataLoader
from utils.data_loading_utils import download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm


class Data:
    def __init__(self, params):
        self.params = params
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class Income(Data):
    def __init__(self, params):
        super().__init__(params)
        self.gen_datasets()
    
    def preprocess(self, data_table):
        encoded_dataset = []

        for col_index in range(self.D):
            # The column over which we compute statistics
            stat_col = data_table[:199523, col_index].reshape(-1, 1)
            non_missing_col = data_table[:, col_index].reshape(-1, 1)

            # Fit on stat_col, transform non_missing_col
            if col_index in self.cat_features:
                fitted_encoder = OneHotEncoder(sparse=False).fit(
                    stat_col)
                encoded_col = fitted_encoder.transform(
                    non_missing_col).astype(np.bool_)

            elif col_index in self.num_features:
                fitted_encoder = StandardScaler().fit(stat_col)
                encoded_col = fitted_encoder.transform(non_missing_col)
            else:
                raise NotImplementedError
            
            encoded_dataset.append(encoded_col)
        
        return np.concatenate(encoded_dataset, axis=1)

    def gen_datasets(self):
        data_path = '/data3/binghui/data/income/income_set_data.pkl'
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train = trainData['V_train'], trainData['S_train']
            self.V_val, self.S_val = valData['V_train'], valData['S_train']
            self.V_test, self.S_test = testData['V_train'], testData['S_train']
        
        else:
            path = Path('../dataset') / 'income'
            data = fetch_openml('Census-income', version=1, data_home=path)

            # target in 'data'
            data_table = data['data']
            if isinstance(data_table, np.ndarray):
                pass
            elif isinstance(data_table, pd.DataFrame):
                data_table = data_table.to_numpy()
            
            self.num_features, self.cat_features = get_num_cat_auto(data_table, cutoff=55)
            self.N = data_table.shape[0]
            self.D = data_table.shape[1]

            data_table = self.preprocess(data_table)

            x_train = data_table[:174582, :-1]
            y_train = data_table[:174582, -1]
            x_val = data_table[174582:199523, :-1]
            y_val = data_table[174582:199523, -1]
            x_test = data_table[199523:, :-1]
            y_test = data_table[199523:, -1]

            np.random.seed(1)   # fix dataset
            V_size, S_size = self.params.v_size, self.params.s_size
            
            self.V_train, self.S_train = get_income_dataset(x_train, y_train, 200000, V_size, S_size)
            self.V_val, self.S_val = get_income_dataset(x_val, y_val, 40000, V_size, S_size)
            self.V_test, self.S_test = get_income_dataset(x_test, y_test, 150000, V_size, S_size)

            trainData = {'V_train': self.V_train, 'S_train': self.S_train}
            valData = {'V_train': self.V_val, 'S_train': self.S_val}
            testData = {'V_train': self.V_test, 'S_train': self.S_test}
            pickle.dump((trainData, valData, testData), open(data_path, "wb"),protocol = 4)

    def get_loaders(self, batch_size, num_workers, shuffle_train=False, get_test=True):
        train_dataset = SetDataset(self.V_train, self.S_train, self.params, is_train=True)
        val_dataset = SetDataset(self.V_val, self.S_val, self.params)
        test_dataset = SetDataset(self.V_test, self.S_test, self.params)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers) if get_test else None
        return train_loader, val_loader, test_loader

class SetDataset(Dataset):
    def __init__(self, V, S, params, is_train=False):
        self.data = V
        self.labels = S
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size

    def __getitem__(self, index):
        V = self.data[index]
        S = self.labels[index]

        S_mask = torch.zeros([self.v_size])
        S_mask[S] = 1
        if self.is_train:
            idxs = (S_mask == 0).nonzero(as_tuple=True)[0]
            neg_S = idxs[torch.randperm(idxs.shape[0])[:S.shape[0] * self.neg_num]]
            neg_S_mask = torch.zeros([self.v_size])
            neg_S_mask[S] = 1
            neg_S_mask[neg_S] = 1
            return V, S_mask, neg_S_mask
        
        return V, S_mask
    
    def __len__(self):
        return len(self.data)

def get_income_dataset(data, y, data_size, v_size, s_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.Tensor(data).to(device)
    img_nums = data.shape[0]

    V_list = []
    S_list = []
    s_index_list = np.argwhere(y>0).squeeze()
    neg_index = np.argwhere(y==0).squeeze()

    cur_size = 0
    pbar = tqdm(total=data_size)
    while True:
        if cur_size == data_size: break
        # s_size = np.random.randint(2, 4)
        s_index = np.random.choice(s_index_list, s_size)
        s_data = data[s_index]
        v_index = np.random.choice(neg_index, v_size)
        V = data[v_index]
        S = np.random.choice(v_size, s_size)
        V[S] = s_data
        S = torch.Tensor(S).type(torch.int64)
        V = torch.tensor(V).cpu()
        
        V_list.append(V)
        S_list.append(S)
        
        cur_size += 1
        pbar.update(1)

    V_list = torch.stack(V_list)
    S_list = torch.stack(S_list)
    pbar.close()
    return V_list, S_list


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
    


