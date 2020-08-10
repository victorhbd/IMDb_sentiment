# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:40:21 2020

@author: u3w1
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack

import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from Imdb_utils import score, Find_Optimal_Cutoff
from Imdb_utils import stem_sentences, Lemmatization

import nltk
stopwords = nltk.corpus.stopwords.words('english') 

df_train = pd.read_csv('train.csv', header=0, na_values='\\N')
df_test = pd.read_csv('test.csv', header=0, na_values='\\N')

columns = ['key_0', 'text', 'titleType', 'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating', 'numVotes']

df_train['0ou1'] = 0
df_train.loc[df_train['sentiment']>5, '0ou1'] = 1

df_test['0ou1'] = 0
df_test.loc[df_test['sentiment']>5, '0ou1'] = 1

df_train['text'] = df_train['text'].str.lower().str.replace('<br />', '')
df_test['text'] = df_test['text'].str.lower().str.replace('<br />', '')

df_train['label'] = df_train['0ou1']
df_test['label'] = df_test['0ou1']

StemOrLemma = ''
if StemOrLemma == 'stem':
    df_train['text'] = df_train['text'].apply(stem_sentences)
    df_test['text'] = df_test['text'].apply(stem_sentences)

elif StemOrLemma == 'lemma':
    lemma = Lemmatization('en_core_web_md')
    df_train[['text', 'pos', 'ent_type', 'vector_norm']] = df_train['text'].apply(lemma.lemma_sentences)
    df_test[['text', 'pos', 'ent_type', 'vector_norm']] = df_test['text'].apply(lemma.lemma_sentences)

# df_train[['text', 'pos', 'ent_type', 'vector_norm']].to_csv('./train_lemma.csv')
# df_test[['text', 'pos', 'ent_type', 'vector_norm']].to_csv('./test_lemma.csv')

X_train = df_train[columns]
y_train = df_train['0ou1'].values
y_test = df_test['0ou1'].values

ngram = 2
tfidf = False
if tfidf:
    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, ngram))
else:
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, ngram))

vectors_train = vectorizer.fit_transform(df_train['text'])
vectors_test = vectorizer.transform(df_test['text'])
vocab_non_ansi = vectorizer.get_feature_names()

#excluimos averageRating pois já indica a avaliação dos outros usuários (poderiamos usar, mas com ressalvas)
feat_from_dataset = []#['titleType', 'startYear', 'endYear', 'runtimeMinutes', 'genres']
if feat_from_dataset:
    if 'genres' in feat_from_dataset:
        genres_train = df_train['genres'].str.get_dummies(sep=',')
        genres_test = df_test['genres'].str.get_dummies(sep=',')
        feat_from_dataset.remove('genres')
        vectors_train = hstack((vectors_train, genres_train), format='csr')
        vectors_test = hstack((vectors_test, genres_test), format='csr')
        vocab_non_ansi = vocab_non_ansi + genres_train.columns.tolist()
    if 'titleType' in feat_from_dataset:
        le = preprocessing.LabelEncoder()
        df_test['titleType'] = df_test['titleType'].fillna('NA')
        le.fit(df_train['titleType'].append(pd.Series(['NA'])))
        df_train['titleType_encoded'] = le.transform(df_train['titleType'])
        df_test['titleType_encoded'] = le.transform(df_test['titleType'])
        feat_from_dataset.remove('titleType')
        feat_from_dataset.append('titleType_encoded')
    
    vectors_train = hstack((vectors_train, df_train[feat_from_dataset]), format='csr')
    vectors_test = hstack((vectors_test, df_test[feat_from_dataset]), format='csr')
    vocab_non_ansi = vocab_non_ansi + feat_from_dataset

def count_list_series(lista_vector_norm):
    lista = []
    for value in lista_vector_norm:
        if value:
            lista.append(np.float(value))
    return pd.Series([sum(lista), sum(lista)/len(lista)])

feat_from_lemma = True
if feat_from_lemma:
    df_train_lemma = pd.read_csv('train_lemma.csv', header=0)
    df_test_lemma = pd.read_csv('train_lemma.csv', header=0)
    
    df_train_lemma['vector_norm'] = df_train_lemma['vector_norm'].str.split(' ')
    df_test_lemma['vector_norm'] = df_test_lemma['vector_norm'].str.split(' ')
    df_train_lemma[['sum_vector_norm','mean_vector_norm']] = df_train_lemma['vector_norm'].apply(count_list_series)
    df_test_lemma[['sum_vector_norm','mean_vector_norm']] = df_test_lemma['vector_norm'].apply(count_list_series)

    vectors_train_pos =  vectorizer.fit_transform(df_train_lemma['pos'])
    vectors_test_pos =  vectorizer.transform(df_test_lemma['pos'])
    vocab_cv_pos = vectorizer.get_feature_names()
    
    vectors_train_ent_type =  vectorizer.fit_transform(df_train_lemma['ent_type'])
    vectors_test_ent_type =  vectorizer.transform(df_test_lemma['ent_type'])
    vocab_cv_ent_type = vectorizer.get_feature_names()
    
    vectors_train = hstack((vectors_train, df_train_lemma[['sum_vector_norm','mean_vector_norm']],
                            vectors_train_pos, vectors_train_ent_type), format='csr')
    vectors_test = hstack((vectors_test, df_test_lemma[['sum_vector_norm','mean_vector_norm']],
                            vectors_test_pos, vectors_test_ent_type), format='csr')
    vocab_non_ansi = vocab_non_ansi + ['sum_vector_norm','mean_vector_norm'] + vocab_cv_pos + vocab_cv_ent_type

x_train = vectors_train.astype(np.float32)
x_test = vectors_test.astype(np.float32)

def removeNonAscii(s):
    string = ''
    for i in s:
        if ord(i)<128:
            string = string + i
        else:
            string = string + '_'
    return string

vocab = []
for word in vocab_non_ansi:
    wd = removeNonAscii(word)
    vocab.append(wd)
     
print('Treinando modelo cv')
CV = 10

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metrics'] = 'binary_error'
params['learning_rate'] = 0.1
params['num_leaves'] = 20
params['min_data_in_leaf'] = 30
params['lambda_l2'] = 0.1

d_train_cv = lgb.Dataset(x_train, label=y_train, feature_name=vocab)
#Cross validation to get best iteration
bst = lgb.cv(params, 
         d_train_cv, 
         nfold = CV,
         shuffle = True,
         stratified = True,
         num_boost_round=10000,
         early_stopping_rounds=100,
         verbose_eval=50,
         seed = 70) 
#get best iteraton
cv_binary_loss = min(bst.get('binary_error-mean'))
best_iter = bst.get('binary_error-mean').index(cv_binary_loss)
#train
m = lgb.train(params, train_set=d_train_cv, num_boost_round=best_iter, verbose_eval=best_iter)
preds_train = m.predict(x_train)
#find optimal cutoff
threshold = Find_Optimal_Cutoff(y_train, preds_train)[0]
#predict
preds_test = m.predict(x_test)
test_acc, test_roc, test_f1 = score(y_test, preds_test, threshold)
#save results
results_train = pd.read_csv('./results/results_train.csv', index_col=0)
results_train = results_train.append({'params':params, 'tfidf':tfidf, 'cv':CV,
                          'feat_from_dataset': feat_from_dataset,'feat_from_lemma': feat_from_lemma,
                          'StemOrLemma': StemOrLemma, 'cv_binary_loss':cv_binary_loss, 
                          'test_acc':test_acc, 'test_roc':test_roc, 'test_f1':test_f1}, ignore_index=True)
results_train.to_csv('./results/results_train.csv')

#%% feat importance  

feat_importance = pd.DataFrame()
feat_importance['features'] = vocab
feat_importance['feat_import_split'] = m.feature_importance(iteration=best_iter)
feat_importance['feat_import_gain'] = m.feature_importance(iteration=best_iter, importance_type='gain')
# feat_importance.sort_values(by=['feat_import_gain'])
# lgb.plot_importance(m)
feat_importance.loc[feat_importance.features == 'good']

#%% shap
import shap
explainer = shap.TreeExplainer(m)
#primeiros que são positivos
shap_values = explainer.shap_values(x_test[0:100])

shap.force_plot(explainer.expected_value[1], shap_values[1][4,:],
                pd.DataFrame(x_test.todense()[4,:], columns = vocab), matplotlib=True)

for i in range(0,5):
    print(i)
    print(df_test.iloc[i]['text'])
    print(preds_test[i])
    print(y_test[i])
    tmp = shap_values[1][i,:].tolist().copy()
    for i in range(0,5):
        value = max(tmp)
        index = tmp.index(value)
        tmp[index] = 0
        print(vocab[index], value)
    for i in range(0,5):
        value = min(tmp)
        index = tmp.index(value)
        tmp[index] = 0
        print(vocab[index], value)   

#últimos que são negativos
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(x_test[-6:])

shap.force_plot(explainer.expected_value[1], shap_values[1][4,:],
                pd.DataFrame(x_test.todense()[4,:], columns = vocab), matplotlib=True)

for i, j in zip(range(0,6), range(24994,25000)):
    print(i)
    print(df_test.iloc[j]['text'])
    print(preds_test[j])
    print(y_test[j])
    tmp = shap_values[1][i,:].tolist().copy()
    for i in range(0,5):
        value = max(tmp)
        index = tmp.index(value)
        tmp[index] = 0
        print(vocab[index], value)
    for i in range(0,5):
        value = min(tmp)
        index = tmp.index(value)
        tmp[index] = 0
        print(vocab[index], value) 

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
index = 12516
shap_values = explainer.shap_values(x_test[index])
print(df_test.iloc[index]['text'])
print(preds_test[index])
print(y_test[index])
tmp = shap_values[1][0,:].tolist().copy()
for i in range(0,5):
    value = max(tmp)
    index = tmp.index(value)
    tmp[index] = 0
    print(vocab[index], value)
for i in range(0,5):
    value = min(tmp)
    index = tmp.index(value)
    tmp[index] = 0
    print(vocab[index], value) 
#%% exemplo de erro com alta certeza - predição negativa

result_test.loc[(result_test.pred < 0.1) & (result_test.target==1)]
index = 73
shap_values = explainer.shap_values(x_test[index])
print(df_test.iloc[index]['text'])
print(preds_test[index])
print(y_test[index])
tmp = shap_values[1][0,:].tolist().copy()
for i in range(0,5):
    value = max(tmp)
    index = tmp.index(value)
    tmp[index] = 0
    print(vocab[index], value)
for i in range(0,5):
    value = min(tmp)
    index = tmp.index(value)
    tmp[index] = 0
    print(vocab[index], value) 
    
#%% tree 0
# m.save_model('light_model', best_iter)
# m = lgb.Booster(model_file='light_model')
lgb.plot_tree(m, tree_index=0)
lgb.create_tree_digraph(m, 0)
