# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:24:59 2020

@author: u3w1
"""

import pandas as pd
import numpy as np

from Imdb_utils import CalcEmbeddingVectorizerSpacy, CalcEmbeddingVectorizer
from Imdb_utils import Find_Optimal_Cutoff, score
import lightgbm as lgb

#%%

df_train = pd.read_csv('train.csv', header=0, na_values='\\N')
df_test = pd.read_csv('test.csv', header=0, na_values='\\N')

columns = ['key_0', 'text', 'titleType', 'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating', 'numVotes']

df_train['0ou1'] = 0
df_train.loc[df_train['sentiment']>5, '0ou1'] = 1

df_test['0ou1'] = 0
df_test.loc[df_test['sentiment']>5, '0ou1'] = 1

df_train['text'] = df_train['text'].str.lower().str.replace('<br />', '')
df_test['text'] = df_test['text'].str.lower().str.replace('<br />', '')

y_train = df_train['0ou1'].values
y_test = df_test['0ou1'].values

#%% ajust text

embeddings = CalcEmbeddingVectorizerSpacy('en_core_web_md', df_train['text'])

import nltk
stopwords = nltk.corpus.stopwords.words('english') 

df_train['text'] = df_train.text.str.replace("[^\w\s]", "")
df_train['text'] = df_train['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])

df_test['text'] = df_test.text.str.replace("[^\w\s]", "")
df_test['text'] = df_test['text'].apply(lambda x: [item for item in x.split() if item not in stopwords])

#%% word embedding spacy

# wv_vector_train = df_train['text'].apply(embeddings.word_average)
# wv_vector_test = df_test['text'].apply(embeddings.word_average)
wv_vector_train = df_train['text'].apply(embeddings.word_sum)
wv_vector_test = df_train['text'].apply(embeddings.word_sum)

wv_vector_train = np.vstack(wv_vector_train)
wv_vector_test = np.vstack(wv_vector_test)

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metrics'] = 'binary_error', #binary_error, auc
params['learning_rate'] = 0.1
params['num_leaves'] = 10
params['min_data_in_leaf'] = 20
params['lambda_l2'] = 0.1

CV = 10

d_train_cv = lgb.Dataset(wv_vector_train, label=y_train)
    
#Cross validation to get best iteration
bst = lgb.cv(params, 
         d_train_cv, 
         nfold = CV,
         shuffle = True,
         stratified = True,
         #metrics = 'None',
         num_boost_round=10000,
         early_stopping_rounds=50,
         verbose_eval=50,
         seed = 70) 

# cv_roc = max(bst.get('auc-mean'))
# best_iter = bst.get('auc-mean').index(max(bst.get('auc-mean')))
# print(cv_roc)
cv_binary_loss = min(bst.get('binary_error-mean'))
best_iter = bst.get('binary_error-mean').index(cv_binary_loss)
print(cv_binary_loss)

m = lgb.train(params, train_set=d_train_cv, num_boost_round=best_iter, verbose_eval=best_iter)

preds_train = m.predict(wv_vector_train)
threshold = Find_Optimal_Cutoff(y_train, preds_train)[0]

preds_test = m.predict(wv_vector_test)
test_acc, test_roc, test_f1 = score(y_test, preds_test, threshold) 

# results_vector = pd.DataFrame()
results_vector = pd.read_csv('./results_vector.csv', index_col=0)
results_vector = results_vector.append({'params':params, 'cv':CV, 'type':'mean_tfidf',
                          'cv_binary_loss':cv_binary_loss, 'embedding': 'spacy',
                          'test_acc':test_acc, 'test_roc':test_roc, 'test_f1':test_f1}, ignore_index=True)
results_vector.to_csv('./results_vector.csv')

#%% resultado do que o modelo tem muita certeza
        
result_test = pd.DataFrame()
result_test['pred'] = preds_test
result_test['target'] = y_test

result_test.loc[result_test.pred > 0.9].hist()
result_test.loc[result_test.pred < 0.1].hist()

print(len(result_test.loc[(result_test.pred > 0.9) & (result_test.target==1)])/len(result_test.loc[(result_test.pred > 0.9)]))
print(len(result_test.loc[(result_test.pred < 0.1) & (result_test.target==0)])/len(result_test.loc[(result_test.pred < 0.1)]))

#%% exemplo de erro com alta certeza - predição positiva
result_test.loc[(result_test.pred > 0.9) & (result_test.target==0)]
index = 15290
df_test.iloc[index]['text']
print(' '.join(df_test.iloc[index]['text']))
print(preds_test[index])
print(y_test[index])

#%% exemplo de erro com alta certeza - predição negativa
result_test.loc[(result_test.pred < 0.1) & (result_test.target==1)]
index = 7716
print(' '.join(df_test.iloc[index]['text']))
print(preds_test[index])
print(y_test[index])

#%%
# https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
# https://nlp.stanford.edu/projects/glove/
# https://fasttext.cc/docs/en/english-vectors.html

mypath= '../../wordEmbeddings/English/'
emb = 'glove.42B.300d.txt'#'enwiki_20180420_300d.txt' #'glove.42B.300d.txt' #'crawl-300d-2M.vec'

embeddings = CalcEmbeddingVectorizer(mypath+emb, df_train['text'], glove=True)

wv_vector_train = df_train['text'].apply(embeddings.word_average)
wv_vector_test = df_test['text'].apply(embeddings.word_average)
wv_vector_train = np.vstack(wv_vector_train)
wv_vector_test = np.vstack(wv_vector_test)

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metrics'] = 'binary_error', #binary_error, auc
params['learning_rate'] = 0.1
params['num_leaves'] = 10
params['min_data_in_leaf'] = 20
params['lambda_l2'] = 0.1

CV = 10

d_train_cv = lgb.Dataset(wv_vector_train, label=y_train)
    
#Cross validation to get best iteration
bst = lgb.cv(params, 
         d_train_cv, 
         nfold = CV,
         shuffle = True,
         stratified = True,
         #metrics = 'None',
         num_boost_round=10000,
         early_stopping_rounds=50,
         verbose_eval=50,
         seed = 70) 

# cv_roc = max(bst.get('auc-mean'))
# best_iter = bst.get('auc-mean').index(max(bst.get('auc-mean')))
# print(cv_roc)
cv_binary_loss = min(bst.get('binary_error-mean'))
best_iter = bst.get('binary_error-mean').index(cv_binary_loss)
print(cv_binary_loss)

m = lgb.train(params, train_set=d_train_cv, num_boost_round=best_iter, verbose_eval=best_iter)

preds_train = m.predict(wv_vector_train)
threshold = Find_Optimal_Cutoff(y_train, preds_train)[0]

preds_test = m.predict(wv_vector_test)
test_acc, test_roc, test_f1 = score(y_test, preds_test, threshold) 

# results_vector = pd.DataFrame()
results_vector = pd.read_csv('./results_vector.csv', index_col=0)
results_vector = results_vector.append({'params':params, 'cv':CV, 'type':'mean',
                          'cv_binary_loss':cv_binary_loss, 'embedding': emb,
                          'test_acc':test_acc, 'test_roc':test_roc, 'test_f1':test_f1}, ignore_index=True)
results_vector.to_csv('./results_vector.csv')
