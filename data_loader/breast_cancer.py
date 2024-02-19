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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Data:
    def __init__(self, params):
        self.params = params
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class BreastCancer(Data):
    def __init__(self, params):
        super().__init__(params)
        file = self.download_breast_cancer()
        self.gen_datasets(file)
    
    def download_breast_cancer(self):
        # Load data from https://www.openml.org/d/4535
        path = Path('../dataset') / 'breastcancer'
        data_name = 'wdbc.data'

        file = path / data_name

        if not file.is_file():
            # download if does not exist
            url = (
                'https://archive.ics.uci.edu/ml/'
                + 'machine-learning-databases/'
                + 'breast-cancer-wisconsin/'
                + data_name)

            download(file, url)
        
        return file
    
    def load_cancer_data(self, file):
        # Read dataset
        data_table = pd.read_csv(file, header=None)
        data_table[1] = data_table[1].apply(lambda x:1 if x=='M' else 0)
        data_table = data_table.to_numpy()

        # Drop id col
        data_table = data_table[:, 1:]

        return data_table

    def gen_datasets(self, file):
        data_path = '../dataset/breastcancer/breat_cancer_set_data.pkl'
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train = trainData['V_train'], trainData['S_train']
            self.V_val, self.S_val = valData['V_train'], valData['S_train']
            self.V_test, self.S_test = testData['V_train'], testData['S_train']
        
        else:
            data_table = self.load_cancer_data(file)

            X_train = data_table[:512, 1:]
            Y_train = data_table[:512, 0]
            x_test = data_table[512:, 1:]
            y_test = data_table[512:, 0]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            x_test  = scaler.transform(x_test)

            x_train = X_train[:398, 1:]
            y_train = Y_train[:398]
            x_val = X_train[398:, 1:]
            y_val = Y_train[398:]
            
            np.random.seed(1)   # fix dataset
            V_size, S_size = self.params.v_size, self.params.s_size
            
            self.V_train, self.S_train = get_cancer_set_dataset(x_train, y_train, 10000, V_size, S_size)
            self.V_val, self.S_val = get_cancer_set_dataset(x_val, y_val, 1000, V_size, S_size)
            self.V_test, self.S_test = get_cancer_set_dataset(x_test, y_test, 1000, V_size, S_size)

            trainData = {'V_train': self.V_train, 'S_train': self.S_train}
            valData = {'V_train': self.V_val, 'S_train': self.S_val}
            testData = {'V_train': self.V_test, 'S_train': self.S_test}
            pickle.dump((trainData, valData, testData), open(data_path, "wb"))

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

def get_cancer_set_dataset(data, y, data_size, v_size, s_size):
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



    


