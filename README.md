# List of projects

## 1) Advanced regression techniques for the prediction of house sale prices

#### In this project, I analyzed a dataset containing 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. It's an expanded and more complex version of the often cited Boston Housing dataset. The target is to predict the sale price of the houses.
### For this project, I decided to compare different regression techniques:

* Simple linear regression using features with highest correlation with target variable
* Linear regression with Ridge/Lasso regularization
* Principal Component Analysis
* Gradient boosting decision trees using XGBoost

The project is at this link: 
[House saleprices project](https://github.com/lbrianza/Data_science/blob/master/House_price_advanced/House%20prices%20-%20advanced%20regression%20techniques.ipynb).
It can also be found in the folder 'House_price_advanced', containing also a .txt file with the description of the 79 features.


## 2) Natural Language Processing - Sentiment analysis on twitter dataset

#### In this project, I analyzed the Sentiment140 dataset, containing 1.6M tweets. Each tweet is labelled as 'positive' or 'negative' based on its sentiment. The target of this project is to build a model able to predict whether a tweet has a 'positive' or 'negative' polarity.
### For this project, I have compared different models:

* Simple Naive Bayes using td-idf
* Convolutional Neural Network
* Recurrent Neural Network using LSTM
* Recurrent Neural Network using LSTM and word embedding using Word2Vec

The project is at this link:
[NLP project](https://github.com/lbrianza/Data_science/blob/master/NLP%20analysis%20sentiment_140/NLP%20-%20analysis%20of%20sentiment140%20dataset.ipynb).
It can also be found in the folder 'NLP analysis sentiment_140'.

## 3) Image classification using Convolutional Neural Networks - Prediction of pneumonia from chest X-ray images comparing different CNN architectures

#### In this project, I analyzed a dataset containing 5783 chest X-ray images of pediatric patients. The goal is to build a model able to predict whether a patient has pneumonia or not.
### In this project, I have compared the performances of 4 different CNN architectures:

* Simple CNN with 5 convolutional layers
* AlexNet
* Transfer learning using VGG-16
* Transfer learning using DenseNet121

The project is at this link:
[CNN project](https://github.com/lbrianza/Data_science/blob/master/Image_classification_chestXray_pneumonia/CNN_comparison_chest_Xray_images.ipynb).
It can also be found in the folder 'Image_classification_chestXray_pneumonia'.


## 4) Speech recognition & multiclass classification problem - Recognition of emotions from audio files using Toronto TESS dataset

#### In this project, I analyzed the Otto  the TESS Toronto dataset, containing around 2800 audio files in WAV format. 
The files contain 200 target words that were spoken in the phrase "Say the word_' by two different voices, and each of the recording portrays one 
of seven different emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral. 
The goal is build a model able to predict the correct emotion of the recording.
### In this project, I have compared the performances of different machine learning algorithms:

* Support Vector Machine
* k-nearest neighbors
* Random forest
* Gradient boosting decision trees

The project is at this link:
[Speech emotion recognition project](https://github.com/lbrianza/Data_science/blob/master/Speech_emotion/Speech_emotion.ipynb).
It can also be found in the folder 'Speech_emotion'.


## 5) Generative Adversarial Network (GAN) - Generation of face images using the CelebA dataset

#### In this notebook, I have used a GAN to generate images that look like those of the celebA dataset, containing more than 200,000 pictures of face images of various celebrities.

The project is at this link:
[GAN project](https://github.com/lbrianza/Data_science/blob/master/GAN%20project/GAN-celebA.ipynb).
It can also be found in the folder 'GAN project'.


## 6) Multiclass classification problem - Comparison of different machine learning algorithms on the categorization of products from the Otto dataset

#### In this project, I analyzed the Otto product dataset, containing 93 features of more than 200,000 products. There are nine categories for all products, 
each of them representing one of the most important product categories (like fashion, electronics, etc.). 
The goal is to build a model able to predict in which category a given product belongs to.
### In this project, I have compared the performances of different machine learning algorithms:

* Logistic regression
* k-nearest neighbors
* Random forest
* Gradient boosting decision trees using XGBoost
* Dimensionality reduction: Principal component analysis and TSNE
* Stacking: combination of different algorithms

The project is at this link:
[Otto project](https://github.com/lbrianza/Data_science/blob/master/Otto_products/Otto_multiclass_classification.ipynb).
It can also be found in the folder 'Otto products'.


## 7) Multivariate time series forecast using Recurrent Neural Networks

#### In this project I have analyzed a multivariate time series representing the total sales of different types of goods from 1993 to 2020 in the US (alcohol, groceries, household appliances and furniture).

The target is to create a model able to predict the future sales (12 months in the future) in the 4 different categories. This has been done by 
using Recurrent Neural Networks (RNN). In particular, two types of architecture have been used:
* LSTM (Long-Short Term Memory)
* GRU (Gated Recurrent Unit)

The project is at this link:
[RNN project](https://github.com/lbrianza/Data_science/blob/master/RNN_project/RNN_sales.ipynb).
It can also be found in the folder 'RNN_project'.

## 8) Train a chess engine

#### In this project I trained a simple ML model to assess a chess position, returning the evaluation number (which tells who has the advantage, white or black, and by how much).

The target is to create a simple 'chess engine' by training a simple ML model, using positions from real games each labelled with the evaluation.

The project is at this link:
[Chess engine](https://github.com/lbrianza/Data_science/blob/master/Chess_engine/Chess_engine.ipynb)
