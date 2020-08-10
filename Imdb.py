# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:52:33 2020

@author: u3w1
"""
from bs4 import BeautifulSoup
import pandas as pd
from requests import get


url = 'http://www.imdb.com/title/tt0453418'
response = get(url)
print(response.text[:500])
response.text.find('app-argument')
print(response.text[316:325])

html_soup = BeautifulSoup(response.text, 'html.parser')
# artist_name_list = html_soup.find(class_='BodyText')


tsv = pd.read_csv('./Desktop/data.tsv', sep='\t', header=0)
aaa = tsv.loc[tsv.knownForTitles.str.contains('tt0453418')]

urls = open('./Desktop/aclImdb/train/urls_pos.txt', 'r').read()
urls_df = urls.split('\n')

# title_akas = pd.read_csv('./Desktop/title_akas.tsv', sep='\t', header=0)
# title_akas.head()
# aaa = title_akas.loc[title_akas.titleId=='tt0100680']
# title_akas.columns
# #'titleId', 'ordering', 'title', 'region', 'language', 'types',
#        # 'attributes', 'isOriginalTitle'
# 
#muitos títulos sem dados ou mais de um título encontrado
# region = []
# language = []
# types = []
# atributes = []
# isOriginalTitle = []
# for i, url in enumerate(urls_df):
#     if len(url) != 48:
#         print(url)
#     index.append(i)
#     id_ = url[26:35]
#     ids.append(id_)
#     temp = title_akas.loc[title_akas.titleId==id_]
#     if len(temp)>1:
#         print('2+ ', i, id_)
#     elif len(temp) == 0:
#         print('0 ', i, id_)
#     else:
#         region.append(temp.region.iloc[0])
#         language.append(temp.language.iloc[0])
#         types.append(temp.types.iloc[0])
#         atributes.append(temp.attributes.iloc[0])
#         isOriginalTitle.append(temp.isOriginalTitle.iloc[0])

title_basics = pd.read_csv('./Desktop/title_basics.tsv', sep='\t', header=0)
title_basics.head()
aaa = title_basics.loc[title_basics.tconst=='tt0100680']
title_basics.columns
# 'tconst', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult',
#        'startYear', 'endYear', 'runtimeMinutes', 'genres'

title_rantings = pd.read_csv('./Desktop/title_rantings.tsv', sep='\t', header=0)
title_rantings.head()
aaa = title_rantings.loc[title_rantings.tconst=='tt0453418']
#       tconst  averageRating  numVotes
# 0  tt0000001            5.6      1627
# 1  tt0000002            6.1       196
# 2  tt0000003            6.5      1317
# 3  tt0000004            6.2       119
# 4  tt0000005            6.1      2102

ids = []
index = []
titleType = []
startYear = []
endYear = []
runtimeMinutes = []
genres = []
averageRatings = []
numVotes = []

# tt0074223 mudou para tt0068444
# tt0127392 mudou para tt0108906
# tt0151604 mudou para tt0097661
# tt0363018 mudou para tt0361856
# tt0935787 mudou para tt0339442
# tt0425671 mudou para tt0410970
# tt0587876 mudou para tt0146793
# tt0849425 mudou para tt0847154

# tt0554587 mudou para tt0235326
# tt0314766 mudou para tt0241251
# tt0428082 mudou para tt0364782
# tt0383735 mudou para tt0312081
# tt0708885 mudou para tt0394911
# tt0384507 mudou para tt0341564
# tt0171736 mudou para tt0117615
# tt0089235 mudou para tt0084035
# tt0765814 tt0499593
# tt0708471 tt0394904
# tt0105011 tt0103987
# tt0177768 tt0171689
# tt0340817 tt0318403
# tt0249376 tt0131857
# tt0303549 tt0165598

valor_anterior = ''

for i, url in enumerate(urls_df):
    if len(url) != 48:
        print(url)
    # index.append(i)
    id_ = url[26:35]
    ids.append(id_)
    temp = title_basics.loc[title_basics.tconst==id_]
    if len(temp)>1:
        print('2+ ', i, id_)
    elif len(temp) == 0:
        # print('0 ', i, id_)
        if valor_anterior == id_:
            new_id = valor_alterado 
        else:
            url = 'http://www.imdb.com/title/' + id_
            response = get(url)
            index_new_id = response.text.find('app-argument')
            new_id = response.text[index_new_id + 27:index_new_id + 36]
        print('De ' + id_ + ' para ' + new_id)
        temp = title_basics.loc[title_basics.tconst==new_id]
        # print(len(temp))
        valor_anterior = id_
        valor_alterado = new_id
    else:
        titleType.append(temp.titleType.iloc[0])
        startYear.append(temp.startYear.iloc[0])
        endYear.append(temp.endYear.iloc[0])
        runtimeMinutes.append(temp.runtimeMinutes.iloc[0])
        genres.append(temp.genres.iloc[0])
    # temp = title_rantings.loc[title_rantings.tconst==id_]
    # if len(temp)>1:
    #     print('-2+ ', i, id_)
    # elif len(temp) == 0:
    #     print('-0 ', i, id_)
    # else:
    #     averageRatings.append(temp.averageRating.iloc[0])
    #     numVotes.append(temp.numVotes.iloc[0])
          

       


# title_crew = pd.read_csv('./Desktop/title_crew.tsv', sep='\t', header=0)
# title_crew.head()
# aaa = title_crew.loc[title_crew.tconst=='tt0453418']
# #directors and writers

# title_episodes = pd.read_csv('./Desktop/title_episodes.tsv', sep='\t', header=0)
# title_episodes.head()
# aaa = title_episodes.loc[title_episodes.tconst=='tt0363018']
# #episodes

# title_principals = pd.read_csv('./Desktop/title_principals.tsv', sep='\t', header=0)
# title_principals.head()
# aaa = title_principals.loc[title_principals.tconst=='tt0453418']
# #jobs - actors and writers