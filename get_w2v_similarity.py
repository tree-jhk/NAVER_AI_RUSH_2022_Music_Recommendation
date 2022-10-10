# -*- coding: utf-8 -*-
import os
import pandas as pd
import sentencepiece as spm
from collections import defaultdict
from gensim.models import Word2Vec as w2v
from collections import Counter
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn
import torch

vocab_size = 24000
method = 'bpe'

model_folder_path = './model'

korean_stopwords = set(['의거하여', '만은 아니다', '여자', '한켠으로는', '힘이', '그런즉', '때문', '등', '공동으로', '점에서 보아', '시키', '할지라도',\
    '7', '사람들', '$', '같은', '되', '그에', '해서는', '여부', '결과에 이르다', '그런 까닭에', '주저하지', '아하', '비추어 보아', '한항목', '너희', \
        '좍좍', '하여야', '”', '예컨대', '게우다', '다른 방면으로', '마저도', '일', '도착하다', '“', '아무도', '정도에 이르다', '퉤', '일단', '비하면',\
            '오로지', '크', '가지', '$', '시간', '쾅쾅', '첫번째로', '대하', '된이상', '다시말하면', '개의치않고', '할줄알다', '이렇다', '중의하나',\
                '하다', '그녀', '요만큼', '다른', '만들', '지', '혹은', '그것', '아니나다를가', '기준으로', '받', '중', '물론', '삐걱거리다', '그래서',\
                    '만약', '와', '말할것도', '꽈당', '에 한하다', '듯하다', '그위에', '관계가 있다', '모두', '따라서', '훨씬', '하기', '소생', '겸사겸사'\
                        '아울러', '&', '안된다', '엉엉', '|', '옆사람', '어떻해', '임에 틀림없다', '어떻게', '만 못하다\t하기보다는', '어느때', '할지언정',\
                            '오자마자', ')', '만큼\t어찌됏든', '남짓', '관한', '얼마나', '중에서', '뿐만', '로', '해도된다', '그렇지 않다면', '네',\
                                '일지라도', '알겠는가', '만 못하다', '︿', '다만', '답다', '마저', '결과에', '지말고', '할수록', '놓', '관해서는', '여',\
                                    '아니다', '조금', '힘입어', '본대로', '무렵', '자마자', '임에', '탕탕', '〉', '고로', '같다', '못하', '일때', '사람', '으로', '앞', '낼', '하는 김에', '이용하여', '이었다', '!', '나오', '점', '어느해', '향하여', '이천팔', '더욱더', '아니라면', '보아', '타다', '이번', '어제', '영', '륙', '할망정', '그중에서', '시각', '또한', '동시에', '쳇', '생각', '시키다', '거니와', '한데', '각종', '의', '의지하여', '부류의', '따름이다', '사', '그래', '말할것도 없고', '누가 알겠는가', '무엇', '를', '틈타', '반대로 말하자면', '시작하여', '어', '지든지', '않다면', '하기 때문에', '그렇지 않으면', '\\', '하여금', '다섯', '왜', '비걱거리다', '끙끙', '바꾸어서', '점에서', '그렇게', '안다', '서술한바와같이', '관계가', '까지도', '보', '비교적', '반대로', '안', '어찌됏어', '하든지', '이천육', '되어다', '6', '아니면', '밖에 안된다', '어떠한', '알았어', '뿐만 아니라', '반드시', '아니', '되어', '없', '관련이', '잇따라', '더군다나', '편이', '얼마든지', '여기', '해요', '어떤것', '도달하다', '메쓰겁다', '견지에서', '구', '안 그러면', '이로 인하여', '야', '‘', '타인', '입각하여', '동안', '참나', '자기', ':', '뚝뚝', '요컨대', '되는', '?', '예하면', '후', '까지', '허걱', '그래도', '저기', '보다더', '딩동', '자신', '생각한대로', '대해서', '이외에도', '함으로써', '헉헉', '쓰여', '@', '저쪽', '어째서', '다수',\
                                        '언젠가', '그렇지만', '당장', '집', '아니라', '비슷하다', '원', '따르', '살', '봐', '하면된다', '인', '가서', '이르기까지', '나머지는', '어떤', '씨', '저희', '와르르', '"', '자기집', '예를', '든간에', '앞의것', '흥', '퍽', '그러한즉', '우리들', '바꿔', '바꾸어말하자면', '이지만', '하게하다', '따지지 않다', '한 후', '그치지', '어찌', '바꾸어말하면', '하면', '0', '그렇지않으면', '만은', '붕붕', '차라리', '정도에', '명', '이', '습니다', '〈', '그러므로', '인젠', '이와같다면', '팔', '따지지', '밖에', '<', '졸졸', '그럼에도 불구하고', '설령', '우선', '잘', '나', '할만하다', '각자', '[', '모르', '당신', '여덟', '하게될것이다', '논하지', '더라도', '몇', '그럼에도', '않고', '즈음하여', '잠깐', '헐떡헐떡', '왜냐하면', '에서', '따라', '만큼', '하기보다는', '함께', '로써', '이천칠', '이천구', '하도록시키다', '일반적으로', '언제', '같이', '하지 않는다면', '향해서', '하느니', '저것만큼', '하지마라', '이 정도의', '어느', '비추어', '것과', '사실', '여러분', '응', '통하', '하면서', '이유만으로', '내', '더욱이는', '제각기', '않으면', '일것이다', '않', '-', '하려고하다', '않는다면', '여전히', '데', '오호', '너희들', '총적으로', '상대적으로 말하자면', ')', '?', '어쨋든', '위해서', '한마디', '게다가', '대로', '하고있었다', '4', '할수있다', '입장에서', '(', '이쪽', '하는것만', '할 따름이다', '버금', '할', '하하', '잠시', '한적이있다', '이르다', '헉', '않기 위해서',\
                                            '기타', '심지어', '줄은모른다', '어느곳', '보이', '즉시', '어느쪽', '즉', '5', '그러', '4', '설사', '들', '예', '조차', '했어요', '바꾸어서 한다면', '#', '자', '우에', '의해서', ']', '그러면', '된바에야', '이렇게되면', '할 생각이다', '어쩔수 없다', '불문하고', '하면 할수록', '삼', '연관되다', '틀림없다', '이다', '해봐요', '》', '어느것', '으로써', '&', '댕그', '의해되다', '구토하다', '대해', '그리고', '전', '아이야', '에', '=', '한', '않기 위하여', '하마터면', '인하여', '이젠', '어디', '아이고', '《', '>', '령', '아홉', '하는바', '으로 인하여', '8', '다음', '에 가서', '둘', '알', '하고 있다', '위에서', '그때', '구체적으로', '해야한다', '결론을', '진짜로', '번', '그렇', '삐걱', '없다', '그저', '아래윗', '위하', '2', '그', '비길수 없다', '까악', '콸콸', '마음대로', '이래', '줄은 몰랏다', '2', '을', '뒤따라', '이유는', '하', '없고', '팍', '못하다', '.', '봐라', '결론을 낼 수 있다', '|', '과', '대해 말하자면', '그리하여', '형식으로', '싶', '。', '칠', '뒤이어', '<', '다음으로', '바꿔 말하면', '그러니까', '둥둥', '하지', '마치', '이상', '요만한 것', '것과 같이', '해도', '7', '아이쿠', '좋', '이와 반대로', '이와 같은', '의해', '가', '때가', '로부터', '년', '어쩔수', '뿐만아니라', '형식으로 쓰여', '하기 위하여', '이와 같다', '때가 되어', '그런데', '쪽으로', '따위', '하지마', '그렇지', '이러이러하다', '대하여', '(', '9', '개', '예를 들면', '요만한걸', '할 줄 안다', '각', '요만한', '이렇', '이 되다', '하나', '하기만', '라 해도', '하도록하다', '솨', '외에도', '오히려', '운운', '않도록', '이러한', '매일', '~', '8', '할뿐', '향하다', '인 듯하다', '—', '한 까닭에', '펄렁', '논하지 않다', '하지만', '바로', '하기만 하면', '또', '*', '3', ';', '딱', '이런', '￥', '그치지 않다', '결국', '일곱', '많은', '해서는 안된다', '및', '미치다', '할 지경이다', '예를 들자면', '이곳', '총적으로 보면', '할때', '하겠는가', '넷', '앗', '전자', '오직', '^', '어기여차', '하는것이', '한 이유는', '어떤것들', '년도', '저', '부류의 사람들', '실로', '기대여', '*', '그들', '전후', '어떻', ',', '말하', '다시 말하자면', '종합한것과같이', '하물며', '위에서 서술한바와같이', '매번', '소인', '적', '얼마간', '무엇때문에', '겨우', '거바', '남들', '줄은', '아무거나', '보면', '에게', '만이', '부터', '이만큼', '이라면', '얼마', '이와', '이때', '연이서', '불구하고', '정도의', '에 대해', "'", '하구나', '1', '말', '있다', '제', '저것', '두번째로', '것', '상대적으로', '어때', '좋아', '하는', '이것', '하는것만 못하다', '할 힘이 있다', '들자면', '응당', '지경이다', '까닭에', '이 외에', '통하여', '까닭으로', '다시', '하고', '어찌됏든', '가까스로', '조차도', '0', '에 있다', '많', '모', '한하다', '매', '관하여', '생각하', '’', '오', '그만이다', '여차', '그러니', '얼마큼', '더', '비길수', '휘익', '우에 종합한것과같이', '4', '해도좋다', '이렇게 많은 것', '과연', '하자마자', '갖고말하자면', '따르는', '하도다', '>', '흐흐', '주저하지 않고', '허허', '이렇게', '이 밖에', '만', '·', '이 때문에', '비로소', '까지 미치다', ';', '우르르', '로 인하여', '%', '월', '오르다', '않기', '김에', '전부', '육', '곧', '여보시오', '와아', '약간', '양자', '총적으로 말하면', '허', '대로 하다', '몰라도', '...', '더구나', '하곤하였다', '아', '하는 편이 낫다', '그렇게 함으로써', '하지 않도록', '+', '두', '이로', '라', '@', '어이', '제외하고', '하더라도', '줄', '설마', '얼마만큼', '되다', '혹시', '말하자면', '정도', '외에', '시초에', '관계없이', '영차', '주', '각각', '아이', '더불어', '한다면 몰라도', '다음에', '할수있어', '토하다', '앞에서', '무슨', '뿐이다', '`', '혼자', '끼익', '어느 년도', '알 수 있다', '그럼', '것들', '그러나', '소리', '에 달려 있다', '바와같이', '생각이다', '(', '그런', '보는데서', '비록', '만약에', '여섯', '수', '+', '않다', '9', '하는것도', '하기는한데', '만이 아니다', '단지', '…', '거의', '윙윙', '속', '이어서', '말하면', '참', '우리', '——', '낫다', '으로서', '이리하여', '바꾸어서 말하면', '사회', '6', '들면', '기점으로', '근거하여', '만일', '습니까', '좀', '지만', '등등', '놀라다', '문제', '아야', '、', '달려', '지금', '관련이 있다', '보드득', '!', '때문에', '셋', '툭', '막론하고', '다소', '한다면', '주룩주룩', '위하여', '이렇게말하자면', '1', '가령', '_', '휴', '%', '같', '고려하면', '와 같은 사람들', '너', '얼마 안 되는 것', '이봐', '누가', '하는것이 낫다', '이럴정도로', '쉿', '아이구', '방면으로', '무릎쓰고', '근거로', '몰랏다', '어떻다', '있', '아니었다면', '쿵', '경우', '어찌하여', '누구', '그에 따르는', '때', '대하면', '5', '하기에', '어찌하든지', '}', '이렇구나'])

english_stopwords = set(['don', 'pages', 'indicate', 'c', 'taken', 'keeps', 'et', 'yet', 'let', 'self', 'take', 'five', "shan't", 'becomes', 'downwards', "we'd", 'willing', 'three', 'able', 'apart', 'seems', 'nearly', 'couldnt', 'showns', 'g', 'system', "wasn't", 'hundred', 'whoever', 'him', 'important', "aren't", 'entirely', 'consider', 'presumably', 'still', 'allow', 'd', 'wasnt', 'plus', 'which', 'tip', 'each', 'ourselves', 'very', 'i', 'secondly', 'f', 'w', 'y', 'hereafter', 'thus', 'whom', 'does', 'got', 'off', 'wouldnt', 'second', 'inc', 'cannot', 'ones', 'upon', 'given', 'us', 'hopefully', 'brief', "won't", "it'll", 'thoughh', 'alone', 'detail', 'definitely', 'seven', 'even', 'auth', 'anyway', 'heres', 'one', 'considering', 'ignored', 'overall', 'r', 'used', 'around', 'successfully', 'ed', 'both', 'sometime', 'ok', "you've", 'specifically', 'tends', 'yes', 'appreciate', 'thru', "they'd", "you're", 'changes', 'throughout', 'empty', 'currently', "we're", "she'd", 'research', 'moreover', 'how', 'between', 'maybe', 'meanwhile', 'hers', 'thin', 'wonder', 'furthermore', 'goes', "where's", 'behind', 'beginnings', 'want', 'aside', 'just', 'getting', 'much', 'besides', 'nevertheless', 'make', 'nine', 'strongly', 'she', 'nay', "c'mon", 'ought', 'edu', 'ex', 'my', 'other', 'sorry', 'l', 'going', "he'd", 'often', 'them', 'begins', 'trying', 'necessary', 'later', 'must', 'any', 'ord', 'soon', 'immediate', "that's", 'specifying', 'throug', "weren't", 'into', 'yourselves', 'wed', 'useful', 'former', 'specify', 'thereafter', 'obviously', 'werent', 'thats', 'see', 'actually', 'com', 'n', 'run', 'biol', 'rd', 'taking', 'became', 'few', 'under', 'next', "'ve", 'did', 'eighty', 'its', "why's", 'find', 'says', 'significant', 'sixty', 'nobody', 'none', 'he', 'their', 'nor', 'importance', 'move', 'lest', 'possibly', 'whether', 'insofar', 'tried', 'since', "isn't", 'try', 'seeming', 'shows', 'ending', 'over', 'hence', 'knows', 'information', 'liked', 'herein', 'such', 'to', 'put', 'necessarily', 'affects', 'look', 'promptly', 'amoungst', 'means', 'down', 'ninety', 'someone', 'six', 'that', 'merely', 'noone', "that'll", 'interest', 'whomever', 'thoroughly', 'gone', 'resulted', 'show', 'the', 'substantially', 'side', 'youd', 'along', 'ours', 'arise', 'this', 'u', 'shown', 'others', 'because', 'sure', 'gave', 'm', 'p', 'vols', 'here', 'anyways', 'pp', 'resulting', "don't", 'somethan', 'associated', 'obtained', 'would', 'yours', 'same', 'date', 'thence', 'gives', 'ca', 'example', 'shed', 'asking', 'away', 'becoming', 'give', 'looking', "who's", 'whim', 'ie', 'need', "i'll", 'contain', "here's", 'therefore', 'part', 'world', "that've", "they've", 'abst', 'sup', 'ups', 'keep', 'www', 'act', 'all', 'exactly', 're', 'million', 'nonetheless', "we've", 'right', 'fifth', 'probably', "it's", 'better', 'can', 'amount', 'hasnt', 'also', 'vol', 'somehow', 'formerly', 'last', 'okay', 'ref', "we'll", 'there', 'whence', 'sometimes', "shouldn't", 'cause', 'some', 'causes', 'tell', 'an', 'tries', 'na', 'id', "doesn't", 'your', 'suggest', 'themselves', 'specified', 'words', 'should', 'call', 'regarding', 'yourself', 'o', 'took', 'doing', 'ff', 'viz', 'indeed', 'accordingly', 'eight', 'certainly', 'que', 'nos', 'different', 'results', 'v', 'well', 'whereafter', 'wherein', 'about', 'mug', 'something', 'two', 'latterly', 'way', "she's", 'was', 'sure\tt', 'below', 'went', 'makes', 's', 'now', 'mostly', 'fifty', 'regardless', 'hid', 'clearly', 'whole', 'anyone', 'saw', 'always', 'namely', "they're", 'km', "he's", 'immediately', 'gets', 'b', 'amongst', 'according', 'onto', 'aren', 'four', 'either', 'various', 'showed', 'those', 'somewhat', 'less', 'found', 'through', 'appear', 'affected', 'whos', 'due', "what's", 'zero', 'lately', 'except', 'sent', 'seem', 'front', 'quickly', 'wants', 'made', 'provides', "'ll", 'gotten', 'new', 'cant', 'relatively', 'these', 'stop', 'inasmuch', 'at', 'never', 'announce', 'similarly', 'by', 'above', 'what', 'best', 'seriously', 'could', 'wherever', 'anyhow', 'sec', 'ltd', 'almost', 'twice', 'contains', 'oh', 'unlike', 'already', 'came', 'kept', 'owing', 'doesn', 'everyone', 'himself', "i'm", 'elsewhere', 'done', 'full', 'however', 'latter', 'giving', 'particular', 'nothing', 'nowhere', 'ask', 'rather', 'near', 'effect', 'everywhere', 'you', 'everybody', 'affecting', 'for', 'especially', 'using', 'primarily', 'respectively', 'indicated', 'unto', 'whatever', 'anybody', 'it', 'thanks', 'thereby', 'another', 'further', 'beforehand', 'or', 'usefully', 'inner', 'related', 'looks', 'widely', 'unfortunately', 'beside', 'me', 'theyd', 'herself', 'please', 'mrs', 'previously', 'sub', 'thanx', 'thereupon', "when's", 'outside', 'briefly', 'indicates', 'may', 'didn', "you'll", 'top', 'wish', 'uses', "a's", 'within', 'hardly', 'hither', 'page', 'appropriate', 'together', 'begin', 'why', 'consequently', 'described', 'get', "hadn't", 'our', 'youre', 'co', 'thorough', 'serious', 'ah', 'every', 'ran', 'enough', 'readily', 'have', 'too', 'ever', 'index', 'per', 'hereby', "mustn't", 'comes', 'containing', 'particularly', "there's", 'said', 'somebody', 'will', 'we', 'whither', 'reasonably', 'welcome', 'forth', 'as', 'like', 'thick', "let's", "he'll", 'vs', 'toward', 'otherwise', 'forty', 'in', 'mg', 'known', 'while', 'once', 'proud', 'out', 'whereupon', 'bill', 'meantime', 'itself', 'having', 'were', 'against', 'more', 'of', 'following', 'q', "t's", 'value', 'name', 'might', 'past', 'been', 'lets', 'sufficiently', 'mine', 'omitted', 'concerning', 'despite', 'possible', "didn't", "haven't", "there've", 'adj', 'needs', 'th', 'say', 'end', 'third', 'sincere', 'j', 'eleven', 'bottom', 'eg', 'whereby', "you'd", 'old', 'theres', 'inward', 'present', 'h', 'fill', 'use', 'shall', 'keep\tkeeps', 'theirs', 'had', 'un', 'be', 'but', 'with', 'ten', 'similar', 'her', 'than', "hasn't", 'follows', 'help', 'x', 'miss', 'im', 'and', 'significantly', 'qv', 'saying', 'do', 'up', "ain't", 'has', 'thereof', 'before', 'instead', 'mr', 'wont', 'from', 'usually', 'own', "i'd", 'not', 'during', 'anywhere', 'apparently', "what'll", "c's", 'know', 'his', 'cry', 'is', 'mill', 'they', 'arent', 'nd', 'whats', 'hed', 'fire', 'likely', 'then', 'where', 'awfully', 'potentially', 'regards', 'thousand', 'if', 'first', 'really', 'when', 'beginning', 'ts', 'e', 'neither', 'k', 'towards', 'etc', 'slightly', 'beyond', 'obtain', 'again', 'wheres', 'added', 'come', 'predominantly', 'theyre', 'largely', 'else', 'corresponding', 'selves', 'am', 'without', 'followed', 'til', 'unlikely', 'poorly', 'twelve', "there'll", "they'll", 'et-al', 'thereto', 'believe', 'allows', 'think', 'seemed', 'approximately', 'fifteen', "i've", 'novel', 'somewhere', 'home', 'back', 'section', 'no', 'quite', 'ml', 'go', 'hi', 'normally', 'little', "she'll", 'howbeit', 'recently', 'thank', 'least', 'whose', 'refs', 'whod', 'computer', 'therein', 'far', 'perhaps', 'greetings', 'a', 'happens', 'several', 'myself', 'course', 'available', 'shes', 'many', 'thou', 'line', 'thered', 'z', 'among', 'afterwards', 'describe', 'anymore', 'become', 'whenever', 'seen', 'are', 'so', 'kg', 'accordance', 'until', 'seeing', 'noted', 'truly', "who'll", 'ours\tourselves', "how's", 'usefulness', 'fix', 'everything', 'mainly', 'though', 'on', 'recent', 'hereupon', 'anything', 'unless', 'hes', "wouldn't", 'whereas', 'across', 'although', 'hello', "couldn't", 'sensible', 'most', 'twenty', 'certain', 'therere', 'non', 'only', 'itd', 'who', 'invention', 'placed', "it'd", 'de', "can't", 'after', 'being', 'via', 'mean', 'con'])

stopwords = korean_stopwords | english_stopwords

# 09:30 -> 시간으로 변환
def convert_time2sec(time:str)->int:
    minute,sec = map(int, time.split(':'))
    return int(60 * minute + sec)

# 박자 계산법: 전처리된 가사 길이 / 음악 재생 시간(초 단위)
# 여기서는 playlist에 속한 모든 박자들의 합과 playlist에 속한 모든 노래의 수로 나눠서 playlist의 평균 박자를 구하고 범주화 시킴
def convert_sec2beat(sum_beat,cnt,beat_category_len):
    x = sum_beat / cnt
    if 0<=x<beat_category_len:
        return str(1111111111)
    elif x<beat_category_len*2:
        return str(2222222222)
    elif x<beat_category_len*3:
        return str(3333333333)
    else:
        return str(0000000000)

# sentencepiece에 넣을 input 생성기
def make_input4tokenizer(ndarray_merged,metadata):
    try:
        # 최종적으로 반환할 토큰 문장들
        sentences = []
    # (1) 플레이리스트별 빈도 수 높은 장르 2개 구하기
        playlist_most_genre = defaultdict(str)
        for i in range(len(ndarray_merged)):
            tmp = []
            for j in range(len(ndarray_merged[i][1])):
                # 장르 전처리
                genre = metadata['album_genre'][str(ndarray_merged[i][1][j])]
                genre = genre.split(',  ')
                tmp.extend(genre)
            most_com_2 = Counter(tmp).most_common(2)
            most_com_2 = sorted(most_com_2,key=lambda x:(x[1],x[0]))
            tmp = []
            for j in range(len(most_com_2)):
                tmp.append(most_com_2[j][0].strip())
            tmp.sort()
            tmp = ' '.join(tmp)
            playlist_most_genre[ndarray_merged[i][0]] = tmp.strip()
        
    # (2) 플레이리스트별 평균 박자 구하기
        playlist_avg_beat = defaultdict(str)
        beat_category_len = 4
        for i in range(len(ndarray_merged)):
            cnt = 0
            sum_beat = 0
            for j in range(len(ndarray_merged[i][1])):
                # 예외 처리: 가사 없는 곡은 평균 박자 구하는 데에 사용하지 않음.
                try:
                    lyric = (metadata['lyric'][str(ndarray_merged[i][1][j])]).lower()
                    lyric = lyric.strip()
                    lyric = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\…》]',' ',lyric).strip()
                    lyric = re.sub('[0-9]',' ',lyric).strip()
                    lyric = lyric.split()
                    tmp = ''
                    for l in lyric:
                        if (l not in stopwords) and len(l) != 1 and len(l) != 0:
                            tmp += (l + ' ')
                    lyric = tmp
                    lyric = lyric.strip()
                    
                    len_lyric = len(lyric)
                    play_time = convert_time2sec(metadata['play_time'][str(ndarray_merged[i][1][j])])
                    cnt += 1
                    sum_beat += len_lyric / play_time
                except:
                    continue
            playlist_avg_beat[ndarray_merged[i][0]] = (convert_sec2beat(sum_beat,cnt,beat_category_len))
        
    # (3) 플레이리스트별 평균 출시 년월 구하기
        playlist_avg_release_date = defaultdict(str)
        for i in range(len(ndarray_merged)):
            sum_year = 0
            sum_month = 0
            for j in range(len(ndarray_merged[i][1])):
                date = metadata['release_date'][str(ndarray_merged[i][1][j])]
                date = date.split('.')
                year = int(date[0])
                month = int(date[1])
                sum_year += year
                sum_month += month
            avg_year = sum_year // ndarray_merged[i][2]
            avg_year_residue = sum_year / ndarray_merged[i][2] - avg_year
            avg_month = round(sum_month / ndarray_merged[i][2])
            determine_avg_month = avg_month + avg_year_residue * 12
            if determine_avg_month >= 12:
                avg_year += 1
                avg_month = int(determine_avg_month - 12)
            else:
                avg_month = int(determine_avg_month)
            playlist_avg_release_date[ndarray_merged[i][0]] = str(avg_year).zfill(4) + (str(avg_month)).zfill(2)
        
    # (4) 플레이리스트별 수록곡들 각각의 가사의 앞 200글자 따고 합치기
        lyric_cut_len = 200
        playlist_lyric = defaultdict(str)
        for i in range(len(ndarray_merged)):
            lyrics = ''
            for j in range(len(ndarray_merged[i][1])):
                # 예외 처리: 가사 없는 곡은 가사 구하는 데에 사용하지 않음.
                try:
                    lyric = (metadata['lyric'][str(ndarray_merged[i][1][j])][:lyric_cut_len]).lower()
                    lyric = lyric.strip()
                    lyric = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\…》]',' ',lyric).strip()
                    lyric = re.sub('[0-9]',' ',lyric).strip()
                    lyric = lyric.split()
                    tmp = ''
                    for l in lyric:
                        if (l not in stopwords) and len(l) != 1 and len(l) != 0:
                            tmp += (l + ' ')
                    lyric = tmp
                    lyric = lyric.strip()
                    lyrics += (lyric + ' ')
                except:
                    continue
            playlist_lyric[ndarray_merged[i][0]] = lyrics.strip()
        # 반환할 sentences list 최종 생성
        for i in range(len(ndarray_merged)):
            sentence = [playlist_lyric[ndarray_merged[i][0]], playlist_most_genre[ndarray_merged[i][0]], playlist_avg_beat[ndarray_merged[i][0]], \
                playlist_avg_release_date[ndarray_merged[i][0]]]
            sentences.append(' '.join(sentence))
    except Exception as e:
        print(e.with_traceback())
        return False
    # 예외 발생 안하면 대형 문자열 list sentences 반환
    # sentences 구성: ['장르 박자정보 출시일자정보 가사정보', ...]
    return sentences, playlist_most_genre, playlist_avg_beat, playlist_avg_release_date, playlist_lyric

class string2vec():
    def __init__(self, train_data, size=200, window=5, min_count=2, workers=20, sg=1, hs=1):
        self.model = w2v(train_data, size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)

    def save_model(self, md_fn):
        self.model.save(md_fn)
        print("word embedding model {} is trained".format(md_fn))

def get_w2v_similarity(ndarray_merged, dict_metadata, dict_train, dict_test):
    # tokenizer_model, w2v_model 저장할 폴더 생성
    model_folder_path = './model'
    os.makedirs(model_folder_path, exist_ok=True)
    
    # spm.SentencePieceTrainer.Train()의 input으로 넣을 txt 생성
    """
    input:
        ndarray_merged : dict 형, merged에 활용되는 ndarray_merged
        metadata : dict 형, metadata
    output:
        sentences 구성: ['장르 박자정보 출시일자정보 가사정보', ...]
    """
    print("Start input sentences generating!")
    
    tokenizer_input_file_path = f'{model_folder_path}/tokenizer_input_{method}_{vocab_size}.txt'
    sentences, playlist_most_genre, playlist_avg_beat, playlist_avg_release_date, playlist_lyric = make_input4tokenizer(ndarray_merged, dict_metadata)
    
    print("Input sentences generating done!")

    print("Start tokenizer model generating!")
    # token 모델 생성
    tokenizer_name = f'{model_folder_path}/tokenizer_{method}_{vocab_size}'
    tokenizer_name_model = f'{model_folder_path}/tokenizer_{method}_{vocab_size}.model'
    
    templates = ' --input={} \
        --pad_id=0 \
        --bos_id=1 \
        --eos_id=2 \
        --unk_id=3 \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage=1.0 \
        --model_type={}'
    os.makedirs(model_folder_path, exist_ok=True)
    with open(tokenizer_input_file_path, 'w', encoding='utf8') as f:
        for sentence in sentences:
            f.write(sentence+'\n')
        
    os.makedirs(model_folder_path, exist_ok=True)
    cmd = templates.format(tokenizer_input_file_path,
                tokenizer_name,    # output model 이름
                vocab_size,# 작을수록 문장을 잘게 쪼갬
                method)# unigram (default), bpe, char
    spm.SentencePieceTrainer.Train(cmd)
    print("SentencePieceTrainer done!")
    
    print('Start generating tokenized_sentences!')
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_name_model)
    
    def get_tokens_from_sentences(sp, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            tokens = sp.EncodeAsPieces(sentence)
            new_tokens = []
            for token in tokens:
                token = token.replace("▁", "")
                if len(token) > 1:
                    new_tokens.append(token)
            if len(new_tokens) > 1:
                tokenized_sentences.append(new_tokens)

        return tokenized_sentences
    
    # 최종 token 문장 생성
    tokenized_sentences = get_tokens_from_sentences(sp, sentences)
    print('Done generating tokenized_sentences!')
    
    print('Start generating w2v model!')
    # w2v 모델 생성
    w2v_name = f'model/w2v_{method}_{vocab_size}.model'
    # w2v 모델 매개변수는 조정하면 됨
    w2v_mod = string2vec(tokenized_sentences, size=200, window=5, min_count=4, workers=8, sg=1, hs=1)
    w2v_mod.save_model(w2v_name)
    w2v_model = w2v.load(w2v_name)
    print('Done generating w2v model!')
    
#######################################################################################################################################
    
    # 유사도 구하는 부분
    print("Start generating w2v embedding results!")
    def get_tokens(sp,sentence):
        tokenized_sentence = []
        tokens = sp.EncodeAsPieces(sentence)
        new_tokens = []
        for token in tokens:
            token = token.replace("▁", "")
            if len(token) > 1:
                new_tokens.append(token)
        if len(new_tokens) > 1:
            tokenized_sentence.append(new_tokens)

        # 2차원 list 반환 -> 실질적으로는 길이 1임.
        return tokenized_sentence

    # playlist_info to vectors
    playlist_info_emb = {}
    for i in tqdm(range(len(ndarray_merged)), desc='putting words in w2v'):
        # 플레이리스트별 meta data들을 토큰화
        pl_lyric = playlist_lyric[ndarray_merged[i][0]]
        pl_lyric_tokens = get_tokens(sp,pl_lyric)
        if len(pl_lyric_tokens):
            pl_lyric_tokens = pl_lyric_tokens[0]
        else:
            pl_lyric_tokens = []
        pl_genre = playlist_most_genre[ndarray_merged[i][0]].split()
        pl_beat = playlist_avg_beat[ndarray_merged[i][0]].split()
        pl_date = playlist_avg_release_date[ndarray_merged[i][0]].split()
        # pl_words : list
        pl_words = pl_lyric_tokens + pl_genre + pl_beat + pl_date
        
        word_embs = []
        for p_word in pl_words:
            try:
                # w2v_model.wv : 단어 벡터를 담은 dict 형 자료
                word_embs.append(w2v_model.wv[p_word])
            except KeyError:
                pass
        if len(word_embs):
            p_emb = np.average(word_embs, axis=0).tolist()
        else:
            p_emb = np.zeros(200).tolist()
        playlist_info_emb[ndarray_merged[i][0]] = p_emb
    
    all_train_ids = [plid for key, plid in dict_train['plid'].items()]
    all_test_ids = [plid for key, plid in dict_test['plid'].items()]

    all_train_ids.sort()
    all_test_ids.sort()
    
    train_ids = []
    train_embs = []
    test_ids = []
    test_embs = []
    
    for pl_id, emb in playlist_info_emb.items():
        if pl_id in all_train_ids:
            train_ids.append(pl_id)
            train_embs.append(emb)
        elif pl_id in all_test_ids:
            test_ids.append(pl_id)
            test_embs.append(emb)
    
    cos = nn.CosineSimilarity(dim=1)
    train_tensor = torch.tensor(train_embs).cuda()
    test_tensor = torch.tensor(test_embs).cuda()
    
    similarity = torch.zeros([test_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64)
    sorted_idx = torch.zeros([test_tensor.shape[0], train_tensor.shape[0]], dtype=torch.int32)
    
    for idx, test_vector in enumerate(tqdm(test_tensor, desc='get cosine sim')):
        output = cos(test_vector.reshape(1, -1), train_tensor)
        sorted_index = torch.argsort(output, descending=True)
        similarity[idx] = output
        sorted_idx[idx] = sorted_index
    
    # test_train_similarity_results = defaultdict(list)
    test_train_similarity_results = dict()
    for i, test_id in enumerate(tqdm(test_ids, desc='get test_train_similarity_results')):
        for j, train_idx in enumerate(sorted_idx[i][:1000]):
            try:
                test_train_similarity_results[test_id].append((train_ids[train_idx], similarity[i][train_idx].item()))
            except:
                test_train_similarity_results[test_id] = [(train_ids[train_idx], similarity[i][train_idx].item())]
    
    return test_train_similarity_results