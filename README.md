# NAVER AI RUSH 2022 Music Recommendation
#### NAVER AI RUSH 2022 Round 2: Music Recommendation 9th
![airush3](https://user-images.githubusercontent.com/97151660/194845726-4442b02d-b8f8-45a2-b091-a0dad785854e.JPG)
## Task
![image](https://user-images.githubusercontent.com/97151660/194845827-f5d573bc-5c9e-4421-8b5b-ab80bff92a7c.png)
VIBE 라는 음악 스트리밍 플랫폼에서 사용자들이 만든 플레이리스트의 수록곡을 맞추는 task입니다.
## Dataset
  train data: 실제 플레이리스트
  meta data: train, test에 담긴 모든 수록곡들에 대한 가사, 가수, 장르 등의 데이터
## Model
- meta data를 잘 활용하기 위해 고민을 제일 많이 했습니다. 대회가 끝난 이후 이에 대해 PM님과 이야기를 해서 저의 meta data 활용보다 더 적절한 방법이 있는 것을 알게 됐습니다.
  - 관련해서 작성한 글: https://velog.io/@tree_jhk/CLOVA-AI-RUSH-CONFERENCE-2022
![image](https://user-images.githubusercontent.com/97151660/194839424-20d3fe02-08af-4a95-8e24-bc6e6fc8781e.png)
- 우선 1개의 latent vector를 갖는 가벼운 autoencoder로 플레이리스트의 수록곡을 예측했습니다. 
![image](https://user-images.githubusercontent.com/97151660/194843165-e03d20bb-d2f5-43d3-a2c5-7d52c7acf6e6.png)
- AutoEncoder 모델의 weight와 bias를 통해 train data와 test data 간의 cos 유사도를 구했습니다.
- Word2Vec을 이용해서 장르, 박자, 발매일, 가사를 concatnate한 문장들을 Word2Vec으로 임베딩하고 train data와 test data 간의 cos 유사도를 구했습니다.
## Recommender
- AE를 이용한 유사도는 test의 수록곡들과 추천된 수록곡들 간의 자카드 유사도를 score로 사용했고
- W2V을 이용한 유사도는 AE의 유사도와 scale을 맞추기 위해, 가중치를 곱하고 score로 사용했습니다.
- 구한 score들을 통해 각 test data별로, top-100 추천 수록곡을 구해서 채워야하는 수록곡 수만큼을 추천했습니다.
## Reference
- YouTube 추천 시스템: https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf
- Spotify 추천 시스템 코드: https://github.com/hojinYang/spotify_recSys_challenge_2018
- kakao arena Melon Playlist Continuation 코드: https://github.com/jjun0127/MelonRec
- Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation: https://pure.mpg.de/rest/items/item_3367572_1/component/file_3367573/content
