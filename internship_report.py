#### index ####
# 1. Subject
# 2. Reference
# 3. Task
# 4. Detail
# 5. Progress
# 6. Plan

#### Subject ####
# My subject is implement the paper, The selfie. That is kind of pretraining. 
# what is pretraining? 
# pretraining is to use part of already trained model when train the model
# what is advantage of Selfie?
# using selfie is improved accuracy and get stable deviation than when not use to selfie

#### Reference ####
# Selfie is reference to model that transformer, BERT, ResNet

#### Task ####
# Selfie is consist of below task
# 1. Data preprocessing
# 2. Image processing
# 3. Masking
# 4. Attention pooling
# 5. Positional embedding
# 6. Prediction
# 7. Finetuning

#### Deatail ####
# 1. Data preprocessing
#  perform zero padding to input data
#  randomized crop to original size
#  slicing to tiles(tile number is intentionally decide)
#
# 2. Image processing
#  each sliced tile is process by P. Then feature of tile is extracted
#  (P is three conv block of ResNet50)
#
# 3. Masking
#
# 4. Attention pooling
#  input  : each feature vector of tile(ie. h)
#  output : single vector(ie. u) 
#  pooling is use transformer layer
#  use attention block of BERT 
#
# 5. Positional embedding
#  use to transformer's position embedding 
#  if input tile of masked out to decoder then  
#
# 6. Prediction
#  add PE of masked out tile with u(ie. v)
#  dot product v with h'1 ~ h'n
#  softmax v*h'1 ~ v*h'n then predict correct tile
#
# 7. Finetuning
#  replace three conv block of train model to learned P 


#### Progress ####
# performing task 3 

#### Plan ####
#
#
#

'''
주제
 - 셀피 구현

서론
 - 셀피는 사전 훈련 기법의 한 종류입니다. 사전 훈련 기법이란 모델을 훈련 시키기 전 미리 훈련을 시킨 사전 훈련 모델을
 훈련 시키려는 모델의 일부분에 이식 후 훈련 하는 방식으로 정확도를 향상 시킬수 있다는 장점을 가지고 있습니다.
 셀피는 사전 훈련 기법에서도 더욱 정확도와 안정적인 편차를 얻을 수 있다는 장점이 있습니다.
 부분적으로 이미지 프로세싱, 마스킹 기법, 어텐션 풀링, 위치 임베딩, 소프트 맥스를 사용합니다.

사용된 기술
 - 셀피에서는 버트, 트랜스포머, 레스넷을 사용했습니다.

단계 설명
1. 입력 이미지를 제로패딩한 뒤 입력이미지 사이즈로 랜덤 크롭, 이를 격자 무늬로 분할
2. 분할된 각각의 격자에 대해 P를 거치면 패치의 특성을 추출, P는 resnet50 모델에서 처음 3개의 합성곱 신경망 블록을 일컫음
3. 추출된 격자의 특성들을 어텐션을 이용한 풀링층(트랜스포머의 인코더)을 거쳐 격자 각각의 위치 임베딩을 가진 싱글벡터 u 를 생성
4. u에 포지션 임베딩을 추가
5. 디코더에서 위치를 예측할 격자를 입력하게 되면 격자의 특성과 v의 유사도를 측정하여 격자의 위치를 예측하게 됨
6. 이를 훈련 시킨 후 P를 resnet에 이식시켜 resnet을 훈련 시킴

진행상황
 - 현재 3단계 진행 중

계획
 - 포지셔널 임베딩 추가 예정


'''q
