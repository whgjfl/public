#### work history #### 
#  6/24 ~ 6/25
# reading and summary paper(Selfie) 
# (link : https://docs.google.com/document/d/19owX6i1E8zsRfNZOx9SiB5ZjHar3vxhRD2kbhHQY2Ac/edit?usp=sharing) 
#
#  6/26 ~ 6/28 
# practice to get used to tensorflow(study CNN using MNIST dataset)
#  - write report that azamat's question about CNN report
#    (study link : https://colab.research.google.com/drive/1UXpv0Na4cH4NjsXxKr7S0nt4qopzHWJ2)
#    (azamat's question report : http://bitly.kr/XnaELM)
#
#  7/1 ~ 7/5
# Practice to get used to tensorflow(implement ResNet)
#  - read and summary journal papers(ResNet)
#  - add content about Using the ResNet package in tensorflow
#   = Slim (link : https://colab.research.google.com/drive/1tRbo0STlCWjVesrw8Laen23ZEL7TQrly)                     
#   = tf.application.contrib.lib (link : http://bitly.kr/XZ0HYV)                 
#
# Implement Selfie  
#  - Checking and reading helpful thing in related journal paper
#    = Unsupervised learning of visual representations by solving jigsaw puzzles(image preprocessing)
#    = BERT(position embedding, pooling)
#                 
#  7/8 ~ 7/12 
# Implement Selfie  
#  - modify searched code for use
#    = Unsupervised learning of visual representations by solving jigsaw puzzles(image preprocessing)
#      (link : https://github.com/bbrattoli/JigsawPuzzlePytorch)
#
#  7/15 ~ 7/17
# after meeting with azamat, Found out that the method of last week was wrong.
# So i accessed other way  
#
# Implement Selfie  
#  - image crop to patch(use image_slicer package)
#  - perform zero-padding and randomized crop to original size 
#  - connect patches of cropped with ResNet50
#    (link : https://colab.research.google.com/drive/1umcyIPbBzvTaRLDc8zXvFuUPE9J9M7oY)
#
#  7/18 ~ 7/19
# Implement Selfie  
#  - laskt week writed code fix(request of azamat)
#  - try to understand structure of BERT, Transformer(use to attention pooling)
#  - try to connect output of ResNet50 with attention pooling
#  - modify transformer source code
#    (link : https://colab.research.google.com/drive/1iT6MvTVdh-YwTly00-YIDMLLrl9PDh3q)
#
#  7/22 ~ 7/24
# Implement Selfie  
#  - change Feedfoward activation function(ReLu) to GeLU
#     (link : https://colab.research.google.com/drive/1UVmIRXXqQeVZnZGZx4sm52ak9hMMgyud)
#  - try connect output of ResNet50 with attention pooling


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
# I study reference paper for implement the Selfie 

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
#  tile is intentinonally masked out(13 tiles masked out is used in Selfie paper)
#  use masking skill of BERT 
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
link = https://colab.research.google.com/drive/1LDxSC1nFRKck7u5AaKVI7xEiFJQerBIh
# performing task 3 

#### Plan ####
#
# add Positional Embedding
#
