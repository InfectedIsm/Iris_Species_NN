#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:30:53 2018

@author: infected
"""
from sklearn.preprocessing import normalize
import tensorflow as tf
import numpy as np
import pandas as pd
import progressbar
#import time 

#import seaborn as sns
#import matplotlib.pyplot as plt
def new_batch(features, labels, batch_size):
	i=0 
	temp_features = np.zeros([batch_size,features[0].shape[0]])
	temp_labels = np.zeros([batch_size,labels[0].shape[0]])
	
	num_datas = len(features)
	sorted_indexes = np.random.randint(0,num_datas-1,batch_size)
	
	for to_next_batch in sorted_indexes :
		temp_features[i] = features[to_next_batch]
		temp_labels[i] = labels[to_next_batch]
		i = i+1
		
	return temp_features,temp_labels

# =============================================================================
# variables d'initialisation
# =============================================================================

hidden_l_neurons = 50

number_of_training = 10000
batch_size = 10

#ci dessous en commentaire, autre méthode pour lire un csv
#data = np.genfromtxt("Iris.csv",skip_header=1, delimiter=',',dtype=str)
iris = pd.read_csv("Iris.csv")

#lorsque l'on lit un fichier csv avec pandas on récupère un objet du type DataFrame, propre à pandas
#La méthode 'values' permet de convertir cet objet en objet numpy
Id = iris['Id'].values
SepalLength = iris['SepalLengthCm'].values
SepalWidth=iris['SepalWidthCm'].values
PetalLength = iris['PetalLengthCm'].values
PetalWidth = iris['PetalWidthCm'].values
Species = iris['Species'].values

#la fonction "set(Species)" permet de lister les valeurs différentes dans une listes
#la fonction "list(set(Species))" permet de convertir le set en list
#print(list(set(Species)))

#features = np.column_stack((SepalLength, SepalWidth, PetalLength, PetalWidth))
features = np.column_stack((PetalLength, PetalWidth))   #les résultats sont bien meilleurs
num_features = np.size(features,1)                     #avec ces 2 features plutôt que les 4 (+14%) 
num_labels = len(set(Species))
num_datas = len(Species)
labels = np.zeros([num_datas,num_labels])

labels[Species=='Iris-setosa']=[1,0,0]
labels[Species=='Iris-versicolor']=[0,1,0]
labels[Species=='Iris-virginica']=[0,0,1]

x_train = features[0:120]
y_train = labels[0:120]

x_test = features[120:150]
y_test = labels[120:150]

# normalisation des features pour une meilleures convergence
#provient de la librarie scikit-learn
normalized_features = normalize(features,axis=0,norm='max')

#dataset = tf.contrib.data.Dataset.from_tensor_slices((dict(features), labels))

tf.reset_default_graph() 

with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32, shape=[None,num_features],name='datas')
	y_ = tf.placeholder(tf.float32, shape=	[None,num_labels],name='labels')
	

with tf.name_scope('first_layer'):
	W1 = tf.Variable(tf.truncated_normal([num_features,hidden_l_neurons], stddev=0.1),name='weights')
	b1 = tf.Variable(tf.truncated_normal([hidden_l_neurons], stddev=0.1),name='weights')
	a1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)

with tf.name_scope('second_layer'):
	W2 = tf.Variable(tf.truncated_normal([hidden_l_neurons,num_labels], stddev=0.1),name='weights')
	b2 = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1),name='weights')
	y = tf.nn.softmax(tf.matmul(a1,W2)+b2)


with tf.name_scope('cross-entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), 
						reduction_indices = [1]))

with tf.name_scope('training'):
	alpha = 0.03
	train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

sess = tf.InteractiveSession()

with tf.name_scope('parameters_initialization'):
	tf.global_variables_initializer().run()
#Chaque tour de boucle, on récupère aléatoirement 100 données d'entrainement
#On fait le training avec une partie du training set : mini-batch gradient
#Avec un seul à chaque fois : stochatstic gradient

with progressbar.ProgressBar(max_value=number_of_training) as bar:
	for actual_iter in range(number_of_training):
		bar.update(actual_iter)
		
		x_batch, y_batch = new_batch(x_train, y_train, batch_size)
		sess.run(train_step, feed_dict={x: x_batch, y_:y_batch})

with tf.name_scope('prediction'):		
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.summary.FileWriter('./logs')
writer.add_graph(tf.get_default_graph())
#graph_def = writer.as_graph_def()
#f = open("./logs/testfile.txt","w") 
#f.write(text_format.MessageToString(graph_def))

print("\nmodel's accuracy (alpha=", alpha, ") :")

print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))