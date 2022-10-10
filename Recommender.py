import numpy as np

from collections import Counter
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

def get_jaccard_score(set1:set, set2:set):
    intersection = set1 & set2
    return len(intersection) / (len(set1) + len(set2) - len(intersection))

def get_new_jaccard_score(set1:set, set2:set):
    intersection = set1 & set2
    return len(intersection) / len(set1)

"""
ae_similarity : defaultdict(list)
    key: testdata_playlist_id / int형
    value: [(testdata의 id와 유사한 traindata id, 그때의 유사도), ... ] / list형
"""
"""
w2v_similarity : defaultdict(list)
    key: testdata_playlist_id / int형
    value: [(testdata의 id와 유사한 traindata id, 그때의 유사도), ...] / list형
"""
def recommender(ae_similarity, w2v_similarity, ndarray_test, ndarray_train, dict_merged, dict_metadata):
    # 최종 추천 결과
    recommend_list = {}
    """
    dict_song_playlist : song_id(key) : playlist_id(value) 구조의 dict
        key: 
            song_id
        value:
            playlist_id
    """
    dict_song_playlist = defaultdict(list)
    for i in range(len(ndarray_train)):
        for song_id in ndarray_train[i][1]:
            dict_song_playlist[song_id] += [ndarray_train[i][0]]
    for key,value in dict_song_playlist.items():
        tmp = set(value)
        dict_song_playlist[key] = tmp
        
    """
    song_freq_data : merged에서의 노래 등장 빈도 수 저장한 dict
        key:
            song_id
        value:
            merged data에서의 빈도 수
    """
    song_freq_data = []
    for property,datalist in dict_merged.items():
        if property == 'tids':
            for i,(pl_id, tids) in enumerate(datalist.items()):
                song_freq_data += tids
    song_freq_data = Counter(song_freq_data)

    """
    dict_playlist_song : 각 playlist별로 보유하는 song list
        key:
            playlist_id
        value:
            song_id list
    """
    dict_playlist_song = dict()
    for i in range(len(ndarray_train)):
        tmp = set(ndarray_train[i][1])
        dict_playlist_song[ndarray_train[i][0]] = list(tmp)
        
    # ae_similarity 기준 topk 추출
    def ae_topk(pl_id, topk):
        # songs[0] : topk에 포함되는 playlist_id        
        # songs[1] : topk에 포함되는 playlist_id와의 유사도
        _topk_playlist = [songs[0] for songs in ae_similarity[pl_id][:topk]]
        _topk_similarity = [songs[1] for songs in ae_similarity[pl_id][:topk]]
        return _topk_playlist, _topk_similarity
    
    # w2v_similarity 기준 topk 추출
    def w2v_topk(pl_id, topk):
        # songs[0] : topk에 포함되는 playlist_id        
        # songs[1] : topk에 포함되는 playlist_id와의 유사도
        _topk_playlist = [songs[0] for songs in w2v_similarity[pl_id][:topk]]
        _topk_similarity = [songs[1] for songs in w2v_similarity[pl_id][:topk]]
        return _topk_playlist, _topk_similarity

    for idx in tqdm(range(len(ndarray_test)), desc='getting recommends'):
        # pl_idx_song_list : ndarray_test[idx]의 수록곡 list
        # playlist_idx : ndarray_test[idx]의 플레이리스트 id
        pl_idx_song_list = deepcopy(ndarray_test[idx][1])
        playlist_idx = deepcopy(ndarray_test[idx][0])
        k = deepcopy(ndarray_test[idx][2])
        tmp4freq = []
        # tmp4freq 결과: playlist ndarray_test[idx]에 포함된 노래들을
        # 포함하고 있는 모든 pl_id들에 대한 list
        # 빈도 기반으로 유사성 판단하기 위함
        for song_id in pl_idx_song_list:
            tmp4freq += dict_song_playlist[song_id]

        # 유사도 기반 추천
        # ndarray_test[idx]와의 유사성을 통한 평점 부여
        rating_dict = defaultdict(lambda: 0)
        
        # topk_ae_playlist : topk에 포함되는 playlist_id        
        # topk_ae_similarity : topk에 포함되는 playlist_id와의 유사도
        topk_ae_playlist, topk_ae_similarity = ae_topk(playlist_idx, 20)
        topk_w2v_playlist, topk_w2v_similarity = w2v_topk(playlist_idx, 50)
        
        # w2v 유사도의 영향력을 ae 유사도의 영향력과 맞추기 위한 가중치 설정
        w2v_weight = np.mean(topk_ae_similarity) / np.mean(topk_w2v_similarity)
        
        deduplicate_dict_song_playlist = defaultdict(set)
        for pl_id in topk_ae_playlist:
            for song_id in dict_playlist_song[pl_id]:
                deduplicate_dict_song_playlist[song_id].add(pl_id)

        # ae로 구한 유사도 활용
        # topk_ae_playlist에는 AE를 기준으로 topk개까지의 유사한 train_playlist_id가 저장되어있다.
        for i, pl_id in enumerate(topk_ae_playlist):
            # song_id는 i번째로 유사한 train_playlist의 수록곡이다.
            for song_id in dict_playlist_song[pl_id]:
                jaccard_score = 0
                # pl_idx_song_list은 현재 추론하려는 test_playlist의 '안 가려진' 수록곡들이 저장되어있다.
                for j in range(len(pl_idx_song_list)):
                    try:
                        # (TARGET) deduplicate_dict_song_playlist[pl_idx_song_list[j]] : 현재 추론하려는 test_playlist의 '안 가려진' j번째 수록곡을 포함하는 playlist들
                        # (COMPARE) deduplicate_dict_song_playlist[song_id] : i번째로 유사한 train_playlist의 수록곡을 포함하는 playlist들
                        # 둘의 '공통' playlist들 개수 / 현재 추론하려는 test_playlist의 '안 가려진' j번째 수록곡을 포함하는 playlist들 개수
                        jaccard_score += get_new_jaccard_score(deduplicate_dict_song_playlist[pl_idx_song_list[j]],\
                                            deduplicate_dict_song_playlist[song_id])
                    except:
                        pass
                rating_dict[song_id] += jaccard_score
        
        # w2v로 구한 유사도 활용
        for i, pl_id in enumerate(topk_w2v_playlist):
            for song_id in dict_playlist_song[pl_id]:
                rating_dict[song_id] += topk_w2v_similarity[i] * w2v_weight
        
        # 내림차순으로 최종 평점 정렬
        rating_dict = sorted(rating_dict.items(),key=lambda x:(x[1],x[0]),reverse=True)
        
        song_recommended = deepcopy(pl_idx_song_list)
        for i, (song_id,score) in enumerate(rating_dict[:100]):
            if len(song_recommended) == k:
                break
            else:
                if song_id not in song_recommended:
                    song_recommended.append(song_id)
        else:
            copy_of_song_freq_data = deepcopy(song_freq_data)
            copy_of_song_freq_data = sorted(copy_of_song_freq_data.items(), key=lambda x:(x[1],x[0]), reverse=True)
            for i in range(len(copy_of_song_freq_data)):
                if len(song_recommended) == k:
                    break
                else:
                    if copy_of_song_freq_data[i][0] not in song_recommended:
                        song_recommended.append(copy_of_song_freq_data[i][0])
                
        recommend_list[playlist_idx] = list(map(int,song_recommended))
    
    return recommend_list