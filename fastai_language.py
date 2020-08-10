# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:19:14 2020

@author: u3w1
"""
import os
import pandas as pd
import numpy as np

from fastai import *
from fastai.text import *
from scipy.spatial.distance import cosine as dist

import torch
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
    
bs=48
torch.cuda.set_device(0)

path = os.path.join('/kaggle', 'input', 'full-imdb')
path

df = pd.read_csv(path+'/df_fastai.csv', index_col = 0)
display(df.head())
print(len(df))

data_lm = TextLMDataBunch.from_csv(path, 'df_fastai.csv', text_cols='text', label_cols='label')

data_lm.show_batch()

learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn_lm.model_dir = '/kaggle/working/'
learn_lm.lr_find()

learn_lm.recorder.plot(skip_end=15)

lr = 1e-3
lr *= bs/48

learn_lm.to_fp16()

learn_lm.fit_one_cycle(1, lr*10, moms=(0.8,0.7))

learn_lm.unfreeze()

learn_lm.fit_one_cycle(12, lr, moms=(0.8,0.7))

#%%salvando modelo gerado

learn_lm.save('fine_tuned_3')

learn_lm.save_encoder('fine_tuned_enc_3')

#%% verificar complementação de frases do modelo

TEXT = "The film is"
N_WORDS = 30
N_SENTENCES = 2

print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

TEXT = "I liked this film because"
N_WORDS = 40
N_SENTENCES = 2

print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

TEXT = "I would not say this movie"
N_WORDS = 40
N_SENTENCES = 2

print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))