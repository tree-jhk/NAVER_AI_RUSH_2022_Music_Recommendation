# -*- coding: utf-8 -*-
import random
import torch.nn as nn

import torch
from tqdm import tqdm
from collections import defaultdict

import os
import torch
import random
import numpy as np
from collections import defaultdict
from torch import nn

def get_latent_vec_emb(ae_model, train_loader, test_loader):
    playlist_embed_weight = []
    playlist_embed_bias = []
    
    # ae_model의 파라미터 접근
    # (name, parameter) 조합의 tuple iterator
    # encoder에서의 weight과 bias 추출
    # latent vector 결과를 구하기 위해 encoder 부분의 weight과 bias가 필요함
    for name, param in ae_model.named_parameters():
        # (param.requires_grad == True) == (파라미터를 학습할 수 있는지)
        if param.requires_grad:
            if name == 'encoder.1.weight':
                playlist_embed_weight = param.data
            elif name == 'encoder.1.bias':
                playlist_embed_bias = param.data

    # train과 test에 대한 잠재 벡터 구함
    ae_embs = dict()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc='train_ae_emb')):
        with torch.no_grad():
            pl_id, x, y, pl_length = batch
            x, y = x.cuda(), y.cuda()
            # latent_res 차원: 은닉층 차원 수 == (1000)
            latent_res = (torch.matmul(y, playlist_embed_weight.T) + playlist_embed_bias).tolist()
            pl_id = list(map(int,pl_id))
            for i in range(len(pl_id)):
                ae_embs[pl_id[i]] = latent_res[i]

    for batch_idx, batch in enumerate(tqdm(test_loader, desc='test_ae_emb')):
        with torch.no_grad():
            pl_id, x, y, pl_length = batch
            x, y = x.cuda(), y.cuda()
            # latent_res 차원: 은닉층 차원 수 == (1000)
            latent_res = (torch.matmul(y, playlist_embed_weight.T) + playlist_embed_bias).tolist()
            pl_id = list(map(int,pl_id))
            for i in range(len(pl_id)):
                ae_embs[pl_id[i]] = latent_res[i]
    return ae_embs
   
"""
ae_embs:
    dict형 자료
        key: pl_id
        testue: pl_id에 대한 latent vector 결과 list -> (1000) 차원
""" 
def get_latent_vec_emb_sim(ae_embs, dict_train, dict_test):
    entire_train_ids = [plid for key, plid in dict_train['plid'].items()]
    entire_test_ids = [plid for key, plid in dict_test['plid'].items()]
    
    print("=============================================================================================")
    print(entire_train_ids)
    print(entire_test_ids)
    
    entire_train_ids.sort()
    entire_test_ids.sort()
    
    train_ids = []
    train_embs = []
    test_ids = []
    test_embs = []

    """
    ae_embs:
        dict형 자료
            key: pl_id
            value: pl_id에 대한 latent vector 결과 list -> (1000) 차원
        train 다음 test 순서로 저장되어 있음(추론할 때 그렇게 추론했음.)
            
    train_ids:
        pl_id가 담겨 있음
        1차원 list -> (49701)
        [1,
        2,
        4,
        6,
        7]
    train_embs:
        train_ids의 요소들인 pl_id들과 같은 순서로, emb벡터(list)가 들어가 있음
        2차원 list -> (49701,1000)
        [[1,2,3,4,5,6,7,9],
        [1,2,3,4,5,6,7,9],
        [1,2,3,4,5,6,7,9],
        [1,2,3,4,5,6,7,9],
        [1,2,3,4,5,6,7,9],]
    test_ids:
        pl_id가 담겨 있음
        1차원 list -> (10000)
    test_embs:
        test_ids의 요소들인 pl_id들과 같은 순서로, emb벡터(list)가 들어가 있음
        2차원 list -> (10000,1000)
    """ 
    for playlist_id, emb in ae_embs.items():
        if playlist_id in entire_train_ids:
            train_ids.append(playlist_id)
            train_embs.append(emb)
        elif playlist_id in entire_test_ids:
            test_ids.append(playlist_id)
            test_embs.append(emb)

    cos = nn.CosineSimilarity(dim=1)
    
    train_tensor = torch.tensor(train_embs).cuda() # train_tensor 차원: (49701,1000)
    test_tensor = torch.tensor(test_embs).cuda() # test_tensor 차원: (10000,1000)

    # test_tensor.shape[0] : 10000
    # train_tensor.shape[0] : 49701
    # similarity, similarity_idx 차원: (10000, 49701)
    similarity = torch.zeros([test_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64)
    sorted_idx = torch.zeros([test_tensor.shape[0], train_tensor.shape[0]], dtype=torch.int32)
    
    """
    test_vector 차원: (1000) == latent vector 차원
    similarity : 각 행(testdata의 playlist들)과 모든 열(traindata의 playlist) 간의 유사도
    sorted_idx : 각 행(testdata의 playlist들)과 모든 열(traindata의 playlist) 간의 유사도 큰 순서로 정렬(내림차순)
    """
    for idx, test_vector in enumerate(tqdm(test_tensor, desc='get cosine sim')):
        """
        test_vector.reshape(1, -1) 차원: (1,1000)
        train_tensor 차원: (49701,1000)
        output 차원: (1,49701) -> test_vec의 train_tensor들과의 유사도 계산
        즉, output은 test_tensor의 특정 playlist가 전체 playlist들 중에서 어떤 playlist와 제일 유사한지를 계산하는 거임.
        """
        output = cos(test_vector.reshape(1, -1), train_tensor)
        # 유사도 큰 순서로 유사도 정렬한 'index'를 반환함
        sorted_index = torch.argsort(output, descending=True)
        similarity[idx] = output
        sorted_idx[idx] = sorted_index

    """
    test_train_similarity_results:
        dict형 자료
            key: testdata의 pl_id
            value: [(testdata의 id와 유사한 traindata id, 그때의 유사도)]
    """
    
    test_train_similarity_results = defaultdict(list)
    for i, test_id in enumerate(tqdm(test_ids, desc='get test_train_similarity_results')):
        # sorted_idx[i][:1000] : testdata의 특정 playlist가 traindata와의 유사도 중에서 가장 큰 1000개
        for j, train_idx in enumerate(sorted_idx[i][:1000]):
            test_train_similarity_results[test_id].append((train_ids[train_idx], similarity[i][train_idx].item()))
            
    return test_train_similarity_results

def get_autoencoder_similarity(ae_model, train_loader, test_loader, dict_train, dict_test):
    # train input 데이터에 대한 잠재 벡터 임베딩 구하기
    """
    ae_emb:
        dict 형
            key: pl_id
            testue: 해당 pl_id에 대한 latent vector 결과
    """
    print("Start generating latent vector embedding results!")
    ae_embs = get_latent_vec_emb(ae_model, train_loader, test_loader)
    
    # 잠재 벡터 임베딩 간의 cos 유사도 계산
    print("Start calculating cosine similarities between each latent vectors!")
    ae_similarity = get_latent_vec_emb_sim(ae_embs, dict_train, dict_test)
    return ae_similarity