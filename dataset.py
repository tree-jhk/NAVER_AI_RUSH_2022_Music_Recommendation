import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import json

class Dataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.num_items = args["num_items"]
        
    def __getitem__(self, index):
        """
        규정상 public하게 공개 불가능한 부분
        """

    def __len__(self):
        return len(self.dataset)

# json 파일 읽기 -> dataloader 이전에 필요
def load_json2ndarray(path, config=False):
    """
    규정상 public하게 공개 불가능한 부분
    """

def load_json2dict(fname):
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)
    return json_obj

# input: 각각 ndarray형
def merge_test_and_train_dict(trainset,testset):
    merged_data = np.concatenate((trainset,testset))
    merged_data = merged_data[merged_data[:,0].argsort()]
    """
    규정상 public하게 공개 불가능한 부분
    """
    dataset = dataset.to_dict()
    return dataset

# input: 각각 ndarray형
def merge_test_and_train_ndarray(trainset,testset):
    merged_data = np.concatenate((trainset,testset))
    merged_data = merged_data[merged_data[:,0].argsort()]
    return merged_data