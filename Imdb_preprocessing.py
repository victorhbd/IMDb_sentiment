# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:52:33 2020

@author: u3w1
"""

# https://ai.stanford.edu/~amaas/data/sentiment/
# https://www.imdb.com/interfaces/

import os
import pandas as pd
from requests import get

def get_id(s):
        url = s[26:35]
        return url

#extrair dados básicos e de ratings dos filmes
def get_dataset(url_urls, url_basics, url_ratings, nm_basics, nm_file):
    
    urls = open(url_urls, 'r').read()
    urls_df = urls.split('\n')
    urls_df = pd.DataFrame(urls_df, columns=['urls'])
    urls_df = urls_df.loc[urls_df.urls != '']
    urls_df['id'] = urls_df['urls'].apply(get_id)
    
    title_basics = pd.read_csv(url_basics, sep='\t', header=0)
    title_ratings = pd.read_csv(url_ratings, sep='\t', header=0)

    df_basics = urls_df.merge(title_basics, left_on='id', right_on='tconst')
    
    #verificar identificadores que não foram relacionados entre as bases
    erros = set(urls_df.id) - set(df_basics.id)
    
    DE, PARA = [], []
    for id_ in erros:
        url = 'http://www.imdb.com/title/' + id_
        response = get(url)
        index_new_id = response.text.find('app-argument')
        new_id = response.text[index_new_id + 27:index_new_id + 36]
        DE.append(id_)
        PARA.append(new_id)
        
    def get_new_id(s):
        try:
            list_index = DE.index(s)
            id_ = PARA[list_index]
        except:
            id_ = s   
        return id_

    urls_df['id_new'] = urls_df['id'].apply(get_new_id)

    df_basics = urls_df.merge(title_basics, left_on='id_new', right_on='tconst', how='left')
    df_basics.to_csv('basics_' + nm_file + '.csv')
    
    df_ratings = urls_df.merge(title_ratings, left_on='id_new', right_on='tconst', how='left')
    df_ratings.to_csv('ratings_' + nm_file + '.csv')
    
    return df_basics, df_ratings

get_dataset('./Desktop/Imdb datasets/aclImdb/train/urls_pos.txt',
            './Desktop/Imdb datasets/title_basics.tsv',
            './Desktop/Imdb datasets/title_ratings.tsv',
            'test_negative')

get_dataset('./Desktop/Imdb datasets/aclImdb/train/urls_neg.txt',
            './Desktop/Imdb datasets/title_basics.tsv',
            './Desktop/Imdb datasets/title_ratings.tsv',
            'test_negative')

get_dataset('./Desktop/Imdb datasets/aclImdb/test/urls_pos.txt',
            './Desktop/Imdb datasets/title_basics.tsv',
            './Desktop/Imdb datasets/title_ratings.tsv',
            'test_negative')

get_dataset('./Desktop/Imdb datasets/aclImdb/test/urls_neg.txt',
            './Desktop/Imdb datasets/title_basics.tsv',
            './Desktop/Imdb datasets/title_ratings.tsv',
            'test_negative')

#extrair dados de texto nos diretórios
def get_text(directory, nm_file):
    comments = pd.DataFrame()
    for filename in os.listdir(directory):
        # print(filename)
        if filename.endswith(".txt"):
            f = open(directory + '/' + filename, encoding="utf8")
            lines = f.read()
            comments = comments.append({'index': filename.split('_')[0],
                                        'sentiment': filename.split('_')[1].split('.')[0],
                                        'text': lines}, ignore_index=True)
            continue
        else:
            continue
        
    comments['index'] = comments['index'].astype(int)
    comments = comments.set_index('index')
    comments.to_csv('text_' + nm_file + '.csv')
    
    return comments

get_text('C:/Users/u3w1/Desktop/Imdb datasets/aclImdb/train/pos', 'train_positive')
get_text('C:/Users/u3w1/Desktop/Imdb datasets/aclImdb/train/neg', 'train_negative')
get_text('C:/Users/u3w1/Desktop/Imdb datasets/aclImdb/test/pos', 'test_positive')
get_text('C:/Users/u3w1/Desktop/Imdb datasets/aclImdb/test/neg', 'test_negative')
#get for unsupervised
get_text('C:/Users/u3w1/Desktop/Imdb datasets/aclImdb/train/unsup', 'train_unsup')

#juntar features no mesmo dataframe
def join_features(nm_file):
    basics = pd.read_csv('basics_' + nm_file + '.csv', index_col=0)
    ratings = pd.read_csv('ratings_' + nm_file + '.csv', index_col=0)
    comments = pd.read_csv('text_' + nm_file + '.csv', index_col=0)

    df = comments.merge(basics, left_on=comments.index, right_on=basics.index)
    df = df.merge(ratings, left_on='key_0', right_on=ratings.index)

    df.to_csv(nm_file+ '.csv')
    
    return df

df_train_pos = join_features('train_positive')
df_train_neg = join_features('train_negative')
join_features('test_positive')
join_features('test_negative')

#juntar dados negativos e positivos
def join_pos_neg(nm_file):
    pos = pd.read_csv(nm_file + '_positive.csv', index_col=0)
    neg = pd.read_csv(nm_file + '_negative.csv', index_col=0)
    
    df = pd.concat([pos, neg])
    df.reset_index(inplace=True)
    
    df.to_csv(nm_file + '.csv')
    
    return df

join_pos_neg('train')
join_pos_neg('test')

#gerar todo o conteúdo texto para trainamento não supervisionado
train_positive = pd.read_csv('text_train_positive.csv', index_col=0)
train_negative = pd.read_csv('text_train_negative.csv', index_col=0)
train_unsup = pd.read_csv('text_train_unsup.csv', index_col=0)

test_positive = pd.read_csv('text_test_positive.csv', index_col=0)
test_negative = pd.read_csv('text_test_negative.csv', index_col=0)

text_train = pd.concat([train_positive, train_negative, train_unsup])
text_train_labeled = pd.concat([train_positive, train_negative])
text_test = pd.concat([test_positive, test_negative])

text_train['text'] = text_train['text'].str.replace('<br />', '')
text_train_labeled['text'] = text_train_labeled['text'].str.replace('<br />', '')
text_test['text'] = text_test['text'].str.replace('<br />', '')

text_train['label'] = 'negative'
text_train.loc[text_train['sentiment']>5, 'label'] = 'positive'
text_train_labeled['label'] = 'negative'
text_train_labeled.loc[text_train['sentiment']>5, 'label'] = 'positive'
text_test['label'] = 'negative'
text_test.loc[text_test['sentiment']>5, 'label'] = 'positive'

text_train = text_train[['label', 'text']]
text_train['text'] = text_train['text'].astype(str)
text_train.to_csv('df_fastai.csv', index=False)

text_train_labeled = text_train_labeled[['label', 'text']]
text_train_labeled['text'] = text_train_labeled['text'].astype(str)
text_train_labeled.to_csv('text_train_labeled.csv', index=False)

text_test = text_test[['label', 'text']]
text_test['text'] = text_test['text'].astype(str)
text_test.to_csv('text_test.csv', index=False)
