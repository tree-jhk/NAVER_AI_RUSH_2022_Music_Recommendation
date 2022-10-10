# -*- coding: utf-8 -*-

from get_autoencoder_similarity import get_autoencoder_similarity
from get_w2v_similarity import get_w2v_similarity
from dataset import *
from torch.utils.data import DataLoader
from Recommender import recommender
import time
from copy import deepcopy

if __name__ == "__main__":
    start_time = time.time()
    sum_time = 0
    """
    규정상 public하게 공개 불가능한 부분
    """

    """
    규정상 public하게 공개 불가능한 부분
    """
    
    print("Basic setups done!")
    
# (0) 각종 데이터 미리 생성하는 단계
    print("PHASE 0 : dataset preparing!")
    
    """
    규정상 public하게 공개 불가능한 부분
    """
    
    # ndarray형 data로
    ndarray_train_m, ndarray_valid = load_json2ndarray(
        train_file_path,
        config=dataset["train_test_split"]
    )
    ndarray_test = load_json2ndarray(test_file_path)
    ndarray_merged = merge_test_and_train_ndarray(ndarray_train_m,ndarray_test)
    ndarray_train = load_json2ndarray(train_file_path)
    
    # dict형 data으로
    dict_train = load_json2dict(train_file_path)
    dict_test = load_json2dict(test_file_path)
    dict_merged = merge_test_and_train_dict(ndarray_train_m,ndarray_test)

    # metadata 불러오기
    metadata_path = '/app/data/meta.json'
    ndarray_metadata = load_json2ndarray(metadata_path)
    dict_metadata = load_json2dict(metadata_path)
    
    # dataloader형 data로
    dataloader_params = config["dataloader"]
    train_loader = DataLoader(Dataset(ndarray_train, config["model"]), **dataloader_params)
    merged_loader = DataLoader(Dataset(ndarray_merged, config["model"]), **dataloader_params)
    valid_loader = DataLoader(Dataset(ndarray_valid, config["model"]), **dataloader_params)
    test_loader = DataLoader(Dataset(ndarray_test, config["model"]))

    print("Dataset setup done!")
    
# (1) AutoEncoder 모델 불러오는 단계
    print("PHASE 1 : AE model preparing!")
    
    """
    규정상 public하게 공개 불가능한 부분
    """
    
    print('ae_model loaded!')

    final_time = time.time()
    elapsed = final_time - start_time
    print(f'Elapsed {elapsed} s')

# (2) w2v model, tokenizer model 생성 및 유사도 계산 단계
    print("PHASE 2 : get w2v based similarity with metadata and train_data and test_data!")
    
    dc_ndarray_merged = deepcopy(ndarray_merged)
    dc_dict_metadata = deepcopy(dict_metadata)
    dc_dict_train = deepcopy(dict_train)
    dc_dict_test = deepcopy(dict_test)
    
    w2v_similarity = get_w2v_similarity(dc_ndarray_merged, dc_dict_metadata,\
        dc_dict_train, dc_dict_test)
    
    print("w2v based similarity calculation done!")
    
    final_time = time.time()
    elapsed = final_time - start_time
    print(f'Elapsed {elapsed} s')
    
# (3) train_data와 test_data 간의 유사도 구하는 단계
    print("PHASE 3 : get similarity between train_data and test_data!")
    
    # ae의 잠재 벡터 간의 유사도 계산
    dc_ae_model = deepcopy(ae_model)
    dc_train_loader = deepcopy(train_loader)
    dc_test_loader = deepcopy(test_loader)
    dc_dict_train = deepcopy(dict_train)
    dc_dict_test = deepcopy(dict_test)
    
    ae_similarity = get_autoencoder_similarity(dc_ae_model, dc_train_loader, dc_test_loader, dc_dict_train, dc_dict_test)
    
    final_time = time.time()
    elapsed = final_time - start_time
    print(f'Elapsed {elapsed} s')
    
# (4) 유사도를 기반으로 song 추천하는 단계
    print("PHASE 4 : final phase, recommending song lists!")
    dc_ae_similarity = deepcopy(ae_similarity)
    dc_w2v_similarity = deepcopy(w2v_similarity)
    dc_ndarray_test = deepcopy(ndarray_test)
    dc_ndarray_train = deepcopy(ndarray_train)
    dc_dict_merged = deepcopy(dict_merged)
    dc_dict_metadata = deepcopy(dict_metadata)
    
    recommend_list = recommender(dc_ae_similarity, dc_w2v_similarity, dc_ndarray_test, dc_ndarray_train, dc_dict_merged, dc_dict_metadata)
    
    """
    규정상 public하게 공개 불가능한 부분
    """
    
    print("Every step done!")
    
    final_time = time.time()
    elapsed = final_time - start_time
    print(f'Elapsed {elapsed} s')