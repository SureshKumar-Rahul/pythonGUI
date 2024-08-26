# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:54:07 2019

@author: jozsef.suto
"""

import numpy as np
import os
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow
from keras.api.datasets import mnist
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras.api.models import load_model
from keras.api.utils import to_categorical
from keras.api.preprocessing import image
#from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.applications import VGG16
from keras.api.callbacks import EarlyStopping
from keras.api.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import copy
import seaborn as sns

def load_mnist_testset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    return train_images, train_labels, test_images, test_labels

def build_neuralnet(classes, input_shape, l0, lmpda):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    #model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), \
    #                       activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, kernel_regularizer=regularizers.l2(lmpda), 
                           activation='softmax'))
    model.compile(optimizers.RMSprop(lr=l0, rho=0.9, epsilon=None, decay=0.0), 
                      loss='categorical_crossentropy', metrics=['accuracy'])   
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def fit_model(model, epochs, traind, trainl, validd=None, validl=None):   #valid_data, valid_labels
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
    mc = ModelCheckpoint('emotiv_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    if validd is None:
        history = model.fit(traind, trainl, 
                            epochs=epochs, batch_size=16,
                            validation_split=0.15,
                            callbacks=[es, mc],
                            verbose=0) 
    else:
        history = model.fit(traind, trainl, 
                            epochs=epochs, batch_size=16,
                            validation_data=(validd, validl), 
                            callbacks=[es, mc], 
                            verbose=0)   
    
                            
    #plot_history(history)
    #best_model = load_model('best_cnn_model.h5')
    #evaluate_model(best_model, shape, test_dir)

    return history

def evaluate_model(model, testd, testl, classes):
    testl = to_categorical(testl, classes)
    #best_model = load_model('best_m.h5')
    test_loss, test_acc = model.evaluate(testd, testl, verbose=0)
    print('Test acc: ', test_acc)
    return test_acc

def k_fold(traind, trainl, testd, testl):
    k = 10
    classes = int(np.max(trainl) + 1)
    num_val_samples = len(traind) // k
    testl = to_categorical(testl, classes)
    
    for r in range(1):
        valid_scores = []
        l0 = 0.00001#10.0**np.random.uniform(-6, -1)
        lmpda = 0.001 #10.0**np.random.uniform(-5, -1)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
        #data = list(zip(traind, trainl))
        #np.random.shuffle(data)
        model = build_neuralnet(classes, (len(traind[0]),), l0, lmpda)
            
        for i in range(k): 
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
            validd = traind[num_val_samples * i : num_val_samples * (i + 1)]
            validl = trainl[num_val_samples * i : num_val_samples * (i + 1)]
            newtraind = np.concatenate([traind[:i * num_val_samples],
                                        traind[(i + 1) * num_val_samples:]], axis=0)
            newtrainl = np.concatenate([trainl[:i * num_val_samples],
                                        trainl[(i + 1) * num_val_samples:]], axis=0)
            newtrainl = to_categorical(newtrainl, classes)
            validl = to_categorical(validl, classes)

            model.fit(newtraind, newtrainl, 
                      epochs=100, batch_size=16, 
                      verbose=1, validation_data=(validd, validl),
                      callbacks=[es])
            #traind[:num_validation_samples * fold] + traind[num_validation_samples * (fold + 1):]        
            #valid_data = np.asarray([i[0] for i in valid])
            #valid_labels = np.asarray([i[1] for i in valid])
            #train_data = np.asarray([i[0] for i in train])
            #train_labels = np.asarray([i[1] for i in train])
            #test_score = evaluate_model(model, testd, testl, classes)
            val_loss, val_acc = model.evaluate(validd, validl, verbose=0)
            valid_scores.append(val_acc)
        final_valid_score = np.average(valid_scores)
        test_loss, test_acc = model.evaluate(testd, testl, verbose=0)     
        print(r, l0, final_valid_score, test_acc)

def keras_random_optimizer(traind, trainl, testd, testl, epochs=1000):
    """Train neural network with random hyper-parameters"""
    classes = int(np.max(trainl) + 1)
    np.random.seed(1000)
    trainl = to_categorical(trainl, classes)
    testl = to_categorical(testl, classes)
    output = []
    best_acc = 0.0

    for i in range(0,20):
        l0 = 10.0**np.random.uniform(-6, -1)
        lmpda = 0.001 #10.0**np.random.uniform(-5, -1)
        model = build_neuralnet(classes, (len(traind[0]),), l0, lmpda)   
        history = fit_model(model, epochs, traind, trainl)
        best_model = load_model('emotiv_model.h5')
        test_loss, test_acc = best_model.evaluate(testd, testl, verbose=0)  
        output.append([l0, test_acc])
        print(i, l0, test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            model.save('best_emotive_model.h5')
            best_history = copy.deepcopy(history) 
    return output, best_history

def keras_net_tester(traindata, trainlabels, testd, testl, epochs=500):
    """Train neural network with random hyper-parameters"""
    v = 0.15
    classes = int(np.max(trainlabels) + 1)
    traind = traindata[0:int(len(traindata) * (1.0 - v))]
    validd = traindata[int(len(traindata) * (1.0 - v)):len(traindata)]
    trainl = trainlabels[0:int(len(trainlabels) * (1.0 - v))]
    validl = trainlabels[int(len(trainlabels) * (1.0 - v)):len(trainlabels)]
    validl = to_categorical(validl, classes)
    trainl = to_categorical(trainl, classes)
    testl = to_categorical(testl, classes)
    l0 = 0.00001
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=200)
    mc = ModelCheckpoint('epoc_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(len(traind[0]),)))
    #model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))
    model.compile(optimizers.RMSprop(lr=l0, rho=0.9, epsilon=None, decay=0.0), 
                  loss='categorical_crossentropy', metrics=['accuracy'])   
    history = model.fit(traind, trainl,
                        epochs=epochs, batch_size=32,
                        #validation_split=0.15,
                        validation_data=(validd, validl),
                        verbose=1, callbacks=[es, mc])
    test_loss, test_acc = model.evaluate(testd, testl, verbose=0) 
    print('Acc: ', test_acc)      
    plot_history(history)
    return model

def plot_history(history):
    plt.plot(history.history['acc'], label='training')
    plt.plot(history.history['val_acc'], label='validation')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend()
    plt.show()
    
def conf(model, X, Y, target_names, normalize=False):
    pred = model.predict(X)
    # Compute confusion matrix
    y_pred = [np.argmax(i) for i in pred]
    y_true = [int(i) for i in Y]
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    sns.heatmap(cm,annot=True,cbar=False)

def plot_confusion_matrix(model, X, Y, target_names,
                          title=None, cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    """
   
    pred = model.predict(X)
    # Compute confusion matrix
    y_pred = [np.argmax(i) for i in pred]
    y_true = [int(i) for i in Y]
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    if title is not None:
        plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')#'\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
# def plot_confusion_matrix(testd, testl, class_names, normalize=False,
#                           title=None, cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
# #    if not title:
# #        if normalize:
# #            title = 'Normalized confusion matrix'
# #        else:
# #            title = 'Confusion matrix, without normalization'
#     best_model = load_model('best_emotive_model.h5')
#     pred = best_model.predict(testd)
#     # Compute confusion matrix
#     y_pred = [np.argmax(i) for i in pred]
#     y_true = [int(i) for i in testl]
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     class_names = [class_names[i] for i in unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=class_names, yticklabels=class_names,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     plt.show()