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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm


class Data:
    def __init__(self, params):
        self.params = params
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class Kick(Data):
    def __init__(self, params):
        super().__init__(params)
        self.gen_datasets()
    
    def preprocess(self, df):
        df['Transmission'] = df['Transmission'].cat.add_categories('Trans_unk')
        df['WheelType'] = df['WheelType'].cat.add_categories('WheelType_unk')
        df['Size'] = df['Size'].cat.add_categories('Trans_unk')
        df['Nationality'] = df['Nationality'].cat.add_categories('Nationality_unk')
        df['TopThreeAmericanName'] = df['TopThreeAmericanName'].cat.add_categories('TopThreeAmericanName_unk')
        df['PRIMEUNIT'] = df['PRIMEUNIT'].cat.add_categories('PRIMEUNIT_unk')
        df['AUCGUART'] = df['AUCGUART'].cat.add_categories('AUCGUART_unk')
        df['Color'] = df['Color'].cat.add_categories('Color_unk')

        train_df = df[:58386]
        test_df = df[58386:]
        imputer=SimpleImputer(strategy='mean')
        imputer.fit(train_df[self.numeric_cols])
        train_df[self.numeric_cols]=imputer.transform(train_df[self.numeric_cols])
        test_df[self.numeric_cols]=imputer.transform(test_df[self.numeric_cols])

        train_df.Transmission.fillna('Trans_unk',inplace=True)
        train_df.WheelType.fillna('WheelType_unk',inplace=True)
        train_df.Nationality.fillna('Nationality_unk',inplace=True)
        train_df.Size.fillna('Trans_unk',inplace=True)
        train_df.TopThreeAmericanName.fillna('TopThreeAmericanName_unk',inplace=True)
        train_df.PRIMEUNIT.fillna('PRIMEUNIT_unk',inplace=True)
        train_df.AUCGUART.fillna('AUCGUART_unk',inplace=True)
        train_df.Color.fillna('Color_unk',inplace=True)

        test_df.TopThreeAmericanName.fillna('TopThreeAmericanName_unk',inplace=True)
        test_df.Size.fillna('Trans_unk',inplace=True)
        test_df.WheelType.fillna('WheelType_unk',inplace=True)
        test_df.Nationality.fillna('Nationality_unk',inplace=True)
        test_df.Transmission.fillna('Trans_unk',inplace=True)
        test_df.PRIMEUNIT.fillna('PRIMEUNIT_unk',inplace=True)
        test_df.AUCGUART.fillna('AUCGUART_unk',inplace=True)
        test_df.Color.fillna('Color_unk',inplace=True)

        encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')
        encoder.fit(train_df[self.categorical_cols])
        encoded_cols=list(encoder.get_feature_names(self.categorical_cols))
        train_df[encoded_cols]=encoder.transform(train_df[self.categorical_cols])
        test_df[encoded_cols]=encoder.transform(test_df[self.categorical_cols])

        scaler=MinMaxScaler()
        scaler.fit(train_df[self.numeric_cols]);
        train_df[self.numeric_cols]=scaler.transform(train_df[self.numeric_cols])
        test_df[self.numeric_cols]=scaler.transform(test_df[self.numeric_cols])

        train_df=train_df[self.numeric_cols+encoded_cols]
        test_df=test_df[self.numeric_cols+encoded_cols]
                
        return train_df, test_df

    def gen_datasets(self):
        data_path = '/data3/binghui/data/kick/kick_set_data.pkl'
        data_home = '/data3/binghui/data/kick'
        if os.path.exists(data_path):
            print(f'load data from {data_path}')
            trainData, valData, testData = pickle.load(open(data_path, "rb"))
            self.V_train, self.S_train = trainData['V_train'], trainData['S_train']
            self.V_val, self.S_val = valData['V_train'], valData['S_train']
            self.V_test, self.S_test = testData['V_train'], testData['S_train']
        
        else:
            x, y = fetch_openml(
                'kick',
                version=1, return_X_y=True, data_home=data_home)
            
            x.drop(['BYRNO','VNZIP1','PurchDate'],axis=1,inplace=True)
            x.drop(['Model','Trim','SubModel','VehYear','WheelTypeID','VNST'],axis=1,inplace=True)
            self.categorical_cols = x.select_dtypes(include = 'category').columns.tolist()
            self.numeric_cols = x.select_dtypes(exclude='category').columns.tolist()
            x["Transmission"].replace("Manual","MANUAL",inplace=True)
            train_df, test_df = self.preprocess(x)

            train_df = train_df.to_numpy()
            test_df = test_df.to_numpy()
            y = pd.to_numeric(y)
            y = y.to_numpy()

            x_train = train_df[:51087]
            y_train = y[:51087]
            x_val = train_df[51087:]
            y_val = y[51087:58386]
            x_test = test_df
            y_test = y[58386:]

            np.random.seed(1)   # fix dataset
            V_size, S_size = self.params.v_size, self.params.s_size
            
            self.V_train, self.S_train = get_income_dataset(x_train, y_train, 150000, V_size, S_size)
            self.V_val, self.S_val = get_income_dataset(x_val, y_val, 20000, V_size, S_size)
            self.V_test, self.S_test = get_income_dataset(x_test, y_test, 40000, V_size, S_size)

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