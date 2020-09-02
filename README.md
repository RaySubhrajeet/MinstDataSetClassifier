# MinstDataSetClassifier

# Dataset:
Extract the data set contained in the minst.7z file. The data contains the original training data and training labels for mnist.  

Extracting  the first 800 images and their labels from the files for this assignment.

Columnize images: Reshape all the images as 784-dimensional vectors.

Train-test split: Split the images and labels into training and testing sets. The
first N images are used as testing images which are queries for K-NN classifier.
The rest of (800 − N) images are used for training. (N is specified as an input
argument.)

# Principal Component Analysis (PCA)

Instead of classifying images in the pixel domain, we usually first project them into a
feature space since raw input data is often too large, noisy, and redundant for analysis.
Dimensionality reduction techniques are used for this purpose. Dimensionality reduction
is the process of reducing the number of dimensions of each data point while preserving
as much essential information as possible. PCA is one of the main techniques of dimensionality reduction. It linearly maps the data into a lower-dimensional space such that
the variance of the data in this lower-dimensional representation is maximized. In this
problem, the objective is to use the scikit-learn PCA package to perform dimensionality
reduction on images extracted from the MNIST dataset. 

•Documentation of the PCA package provided by scikit-learn (http://
scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

• Use the scikit-learn PCA package to specify a full SVD solver to build your PCA
model (i.e. svd solver=“full”). (Otherwise, the results may not be consistent if the
randomized truncated SVD solver is used in the scikit-learn PCA package.)

• You need to perform dimensionality reduction on both training and testing sets to
reduce the dimension of data from 784 to D. (D is specified as an input argument.)

• Compute the PCA transformations for both train and test data, by fitting the PCA
model only on the training set.

# K-Nearest Neighbors (K-NN)
K-nearest neighbors algorithm (K-NN) is a non-parametric method used for classification.
A query object is classified by a majority vote of the K closest training examples
(i.e. its neighbors) in the feature space. In this problem, the objective is to implement
a K-NN classifier to perform image classification given the image features obtained by
PCA. You need to:
• Implement a K-Nearest Neighbors classifier to predict the class labels of testing
images. Make sure you use the inverse of Euclidean distance as the metric for
the voting. In other words, each neighbor ni
, where i = 1, ..., K, represented as a
vector, contributes to the voting with the weight of 1/
||x−ni||2
, where x is a queried
vector. (K is specified as an input argument.)

# command
 $ python MinstDatasetCLassifier.py K D N PATH_TO_DATA_DIR
 
 where K,specifies the  K nearest neighbor,
       D, is dimensions to be reduced by PCA ie to get D principal component
       N is the split between test and training samples .So test sample size=N, training sample size is 800-N
       PATH_TO_DATA_DIR is the folder path where the dataset is present
