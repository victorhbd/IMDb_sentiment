# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:40:42 2020

@author: u3w1
"""
import os
import pandas as pd
import numpy as np

from fastai import *
from fastai.text import *
import pandas as pd
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from Imdb_utils import score

import torch
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')
    
path = os.path.join('/kaggle', 'input', 'databunch')
data_lm = load_data(path, 'lm_databunch', bs=48)

path = os.path.join('/kaggle', 'input', 'dataset-csv')
data_clas = TextClasDataBunch.from_csv(path, 'text_train_labeled.csv', test='text_test.csv', vocab=data_lm.train_ds.vocab, bs=32)

data_clas.show_batch()

learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()

learn_c.model_dir = '/kaggle/input/encoder3'
learn_c.load_encoder('fine_tuned_enc_3')
learn_c.freeze()

learn_c.model_dir = '/kaggle/output/working'
learn_c.lr_find()

learn_c.recorder.plot()

learn_c.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn_c.save('1st')

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn_c.save('2nd')

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn_c.save('3rd')

learn_c.unfreeze()
learn_c.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn_c.save('final_class')

preds = learn_c.get_preds(ds_type=DatasetType.Test, ordered=True)
p = preds[0].cpu().data.numpy()

pred = pd.DataFrame()
pred['prediction'] = p[:,1]
#preds = pd.read_csv('submission_3.csv', index_col=0)

text_test = pd.read_csv('text_test.csv')

text_test['target'] = 0
text_test.loc[text_test['label']=='positive', 'target'] = 1

test_acc, test_roc, test_f1 = score(text_test['target'], preds, 0.5)

#%% resultado do que o modelo tem muita certeza
        
result_test = pd.DataFrame()
result_test['pred'] = preds.prediction
result_test['target'] = text_test['target']

result_test.loc[result_test.pred > 0.9].hist()
result_test.loc[result_test.pred < 0.1].hist()

print(len(result_test.loc[(result_test.pred > 0.9) & (result_test.target==1)])/len(result_test.loc[(result_test.pred > 0.9)]))
print(len(result_test.loc[(result_test.pred < 0.1) & (result_test.target==0)])/len(result_test.loc[(result_test.pred < 0.1)]))

#%% top losses
interp = TextClassificationInterpretation.from_learner(learn_c) 
interp.show_top_losses(20)

#%% interpret errors - positive
result_test.loc[(result_test.pred > 0.9) & (result_test.target==0)]
interp.show_intrinsic_attention(text_test.iloc[12517]['text'])
interp.show_intrinsic_attention(text_test.iloc[24813]['text'])

#%%interpret errors - negative
result_test.loc[(result_test.pred < 0.1) & (result_test.target==1)]
interp.show_intrinsic_attention(text_test.iloc[144]['text'])
interp.show_intrinsic_attention(text_test.iloc[287]['text'])

#%%
result_test.loc[(result_test.pred < 0.1) & (result_test.target==0)]
text_test.iloc[12505]['text']
