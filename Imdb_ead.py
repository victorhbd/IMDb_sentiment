# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:35:31 2020

@author: u3w1
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
import nltk

#%% read 

df_train = pd.read_csv('train.csv', header=0, na_values='\\N')
df_test = pd.read_csv('test.csv', header=0, na_values='\\N')

columns = ['key_0', 'text', 'titleType', 'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRating', 'numVotes', 'sentiment']

df_train['0ou1'] = 0
df_train.loc[df_train['sentiment']>5, '0ou1'] = 1

df_test['0ou1'] = 0
df_test.loc[df_train['sentiment']>5, '0ou1'] = 1

#%% word cloud

stopwords = nltk.corpus.stopwords.words('english')  

text = ' '.join(df_train['text'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000, collocations=False).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words')
plt.axis("off")
plt.show()

df_train['text'] = df_train['text'].str.lower().str.replace('<br />', '')
df_test['text'] = df_test['text'].str.lower().str.replace('<br />', '')
stopwords.append('movie')
stopwords.append('film')

text = ' '.join(df_train.loc[df_train['0ou1'] == 1]['text'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000, collocations=False).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words')
plt.axis("off")
plt.show()

text = ' '.join(df_train.loc[df_train['0ou1'] == 0]['text'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000, collocations=False).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words')
plt.axis("off")
plt.show()

text = ' '.join(df_test.loc[df_test['0ou1'] == 1]['text'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000, collocations=False).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words')
plt.axis("off")
plt.show()

text = ' '.join(df_test.loc[df_test['0ou1'] == 0]['text'].str.lower().values)
wordcloud = WordCloud(max_font_size=None, stopwords=stopwords, background_color='white',
                      width=1200, height=1000, collocations=False).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words')
plt.axis("off")
plt.show()

#%% verificando palavras like e good dos comentários negativos

like_negativos = df_train.loc[(df_train['0ou1'] == 0) & (df_train['text'].str.contains(' like '))]

for i in range(0,5):
    index = like_negativos.iloc[i]['text'].find(' like ')
    print(like_negativos.iloc[i]['text'][index-30:index+30])

good_negativos = df_train.loc[(df_train['0ou1'] == 0) & (df_train['text'].str.contains(' good '))]

for i in range(0,5):
    index = good_negativos.iloc[i]['text'].find(' good ')
    print(good_negativos.iloc[i]['text'][index-50:index+50])

#%% columns

columns = ['key_0', 'text', 'titleType', 'startYear', 'endYear', 'runtimeMinutes', 'genres', 'averageRatings', 'numVotes', 'sentiment']

for i, col in enumerate(set(columns) - set(['text', 'key_0'])):
    print(col)
    # plt.rcParams["figure.figsize"] = [6.4, 4.8]
    if df_train[col].dtype == object:
        cat_unique = len(df_train[col].unique())
        if cat_unique > 15:
            # plt.rcParams["figure.figsize"] = [8+cat_unique*0.01, 5+cat_unique*0.15]
            plt.figure(i)
        else:
            plt.figure(i)
        # plt.rcParams["xtick.labelsize"] = 7
        sns_plot = sns.countplot(y=df_train[col])
    elif df_train[col].dtype == np.float64:
        plt.figure(i)
        # sisbrocas[col] = sisbrocas[col].astype(np.float32)
        sns_plot = sns.distplot(df_train[col] , color="red", vertical=True)
    elif df_train[col].dtype == np.int64:
        plt.figure(i)
        # sisbrocas[col] = sisbrocas[col].astype(np.int32)
        sns_plot = sns.distplot(df_train[col] , color="blue", vertical=True)
    else:
        print('Sem gráfico')
        
    # props = {'boxstyle': 'round', 'facecolor':'wheat', 'alpha': 0.5}
    # plt.text(6, 4, text, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=props)
    figure = sns_plot.get_figure()   
    figure.savefig('./imdb_plot/'+col+'.png', bbox_inches='tight')
    plt.close()

#%% pairplot
    
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_test])
columns.append('dataset')
sns.pairplot(df[columns], hue="dataset")

#%% histogramas de treinamento e teste

#filling NA to plot
df_test['titleType'] = df_test['titleType'].fillna('NA')
#columns to plot
columns = ['titleType', 'startYear', 'endYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'sentiment']

#interate on columns
for col in columns:
    plt.hist(df_train[col], alpha=0.5, label='train')
    plt.hist(df_test[col], alpha=0.5, label='test')
    plt.legend(loc='upper left')
    plt.xticks(rotation='vertical')
    plt.title(col)
    plt.show()

#%% genres
    
genres_train = df_train['genres'].str.get_dummies(sep=',')
genres_test = df_test['genres'].str.get_dummies(sep=',')

genres_train['dataset'] = 'train'
genres_test['dataset'] = 'test'

genres = pd.concat([genres_train, genres_test])

fig, axes = plt.subplots(7, 4, figsize=(20, 30))
for ax, col in zip(axes.flat, genres.columns[0:-1]):
    sns_plot = sns.countplot(y = col, hue='dataset', data=genres, ax=ax)
    

#%% sentiment vs averageRating

sns.distplot(y = 'sentiment', hue='averageRating', data=df_train)

plt.scatter(df_train.loc[df_train['0ou1'] == 1]['averageRating'], df_train.loc[df_train['0ou1'] == 1]['sentiment'], label='positive')
plt.scatter(df_train.loc[df_train['0ou1'] == 0]['averageRating'], df_train.loc[df_train['0ou1'] == 0]['sentiment'], label='negative')
plt.title('Distribuição dos sentimentos pelos ratings dos títulos')
plt.xlabel('averageRating')
plt.ylabel('sentiment')
plt.legend(loc='middle right')

#%% sentiment vs numVotes

plt.scatter(df_train.loc[df_train['0ou1'] == 1]['numVotes'], df_train.loc[df_train['0ou1'] == 1]['sentiment'], label='positive')
plt.scatter(df_train.loc[df_train['0ou1'] == 0]['numVotes'], df_train.loc[df_train['0ou1'] == 0]['sentiment'], label='negative')
plt.title('Distribuição dos sentimentos pelos ratings dos títulos')
plt.xlabel('numVotes')
plt.ylabel('sentiment')
plt.legend(loc='upper right')