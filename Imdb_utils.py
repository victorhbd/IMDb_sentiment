# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:43:23 2020

@author: u3w1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

import spacy
import nltk
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

def score(y_true, y_pred, threshold=0.5):
    
    roc_score = roc_auc_score(y_true, y_pred)
    
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.draw()
    
    
    y_pred = y_pred > threshold
    y_pred = y_pred.astype(int)
    
    plt.figure()
    conf_m = confusion_matrix(y_true, y_pred)#, normalize='true')
    sns.heatmap(conf_m, annot=True, cmap="YlGnBu", fmt='g')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.draw()
    # print('F1 Score: ' + str(f1_score(y_true, y_pred)))
    # precision_score(y_true, y_pred)
    # recall_score(y_true, y_pred)
    
    print('Acurácia: ' + str(accuracy_score(y_true, y_pred)))
    print('AUC: ' + str(roc_score))
    print()
    print(classification_report(y_true, y_pred))
    
    return accuracy_score(y_true, y_pred), roc_score, f1_score(y_true, y_pred)


#%% function to find optimal cutoff from roc_curve

def Find_Optimal_Cutoff(target, predicted):

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

#%% Lematization

class Lemmatization(object):
    
    def __init__(self, model):
        self.nlp = spacy.load(model)
        
    def lemma_sentences(self, sentence):
        
        pos = ""
        lemma = ""
        ent_type = ""
        vector_norm = ""
        # tag = ""
        # cluster = ""
        for token in self.nlp(sentence):
            # text += token.text + " "
            pos += token.pos_ + " "
            lemma += token.lemma_ + " "
            ent_type += token.ent_type_ + " "
            vector_norm += str(token.vector_norm) + " "
            # tag += token.tag_ + " "
            # cluster += str(token.cluster) + " "
        return pd.Series([lemma, pos, ent_type, vector_norm])#, tag, cluster
        
#%% Stem
def stem_sentences(sentence):
    stemmer = nltk.stem.PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

#%% word average and sum - GENSIM

class CalcEmbeddingVectorizer:

    def __init__(self, word_model, serie, glove=False):
        if not glove:
            self.embeddings = KeyedVectors.load_word2vec_format(word_model, binary=False, unicode_errors="ignore")
        
        else:
            tmp_file = get_tmpfile("test_word2vec.txt")
            _ = glove2word2vec(word_model, tmp_file)
            self.embeddings = KeyedVectors.load_word2vec_format(tmp_file)
    
    def word_average(self, sent):
        mean=[]
        for word in sent:
            try:
                mean.append(self.embeddings.get_vector(word))
            except:
                continue
        if not mean:
            return np.zeros(self.embeddings.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
    
    def word_sum(self, sent):
        sum_=[]
        for word in sent:
            try:
                sum_.append(self.embeddings.get_vector(word))
            except:
                continue
        if not sum_:
            return np.zeros(self.embeddings.vector_size)
        else:
            sum_ = np.array(sum_).sum(axis=0)
            return sum_
        
#%% word average and sum - SPACY

class CalcEmbeddingVectorizerSpacy:

    def __init__(self, word_model, serie):
        self.embeddings = spacy.load('en_core_web_md')
    
    def word_average(self, sent):
        mean=[]
        for word in sent:
            try:
                mean.append(self.embeddings.vocab[word].vector)
            except:
                continue
                
        if not mean:
            return np.zeros(300)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
    
    def word_sum(self, sent):
        sum_=[]
        for word in sent:
            try:
                sum_.append(self.embeddings.vocab[word].vector)
            except:
                continue
        if not sum_:
            return np.zeros(300)
        else:
            sum_ = np.array(sum_).sum(axis=0)
            return sum_