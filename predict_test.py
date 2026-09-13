"""Membuat prediksi test.csv dari model website.
Ekstrak test.zip ke ../data/test/ dan letakkan sample_submission.csv di ../data/.
"""
from pathlib import Path
import sys, pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_utils import RetinaMultiLabelPredictor
p=RetinaMultiLabelPredictor(ROOT/'models'/'retina_multilabel_cnn.pt')
s=pd.read_csv(ROOT/'data'/'sample_submission.csv');label_cols=[c for c in s.columns if c!='filename']
for i,fn in enumerate(s.filename):
    raw=(ROOT/'data'/'test'/fn).read_bytes();r=p.predict_bytes(raw,fn);score={x['key']:x['value']/100 for x in r['probabilities']}
    for c in label_cols:s.loc[i,c]=score[c]
s.to_csv(ROOT/'prediction_submission.csv',index=False);print('saved',ROOT/'prediction_submission.csv')
