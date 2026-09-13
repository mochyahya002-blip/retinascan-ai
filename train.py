"""Training ulang CNN multi-label.
Letakkan train.csv di ../data/train.csv dan gambar di ../data/train/<filename>.
Script ini sengaja sederhana agar mudah direproduksi; untuk eksperimen serius tambahkan
augmentasi, cross-validation, tuning arsitektur, dan external validation.
"""
import json, os, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; MODEL_DIR=ROOT/'models'; MODEL_DIR.mkdir(exist_ok=True)
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
LABELS=['opacity','diabetic retinopathy','glaucoma','macular edema','macular degeneration','retinal vascular occlusion','normal']
SIZE=96; EPOCHS=int(os.getenv('EPOCHS','10')); BATCH=int(os.getenv('BATCH_SIZE','64'))
class DS(Dataset):
    def __init__(self,df,tf):self.df=df.reset_index(drop=True);self.tf=tf
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];im=Image.open(DATA/'train'/r.filename).convert('RGB');return self.tf(im),torch.tensor(r[LABELS].values.astype('float32'))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def b(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        self.features=nn.Sequential(b(3,16),b(16,32),b(32,64),b(64,96));self.pool=nn.AdaptiveAvgPool2d(1);self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,7))
    def forward(self,x):return self.head(self.pool(self.features(x)))
def main():
    df=pd.read_csv(DATA/'train.csv');idx=np.arange(len(df));np.random.default_rng(SEED).shuffle(idx);cut=int(.8*len(idx));tr,va=idx[:cut],idx[cut:]
    train_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    val_tf=transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    tr_dl=DataLoader(Subset(DS(df,train_tf),tr),batch_size=BATCH,shuffle=True,num_workers=2);va_dl=DataLoader(Subset(DS(df,val_tf),va),batch_size=BATCH,num_workers=2)
    pos=torch.tensor(df.iloc[tr][LABELS].sum().values.astype('float32'));pw=(len(tr)-pos)/torch.clamp(pos,min=1)
    m=Net();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss(pos_weight=pw);best=-1;history=[]
    for ep in range(1,EPOCHS+1):
        m.train();total=0
        for x,y in tr_dl:opt.zero_grad();z=m(x);loss=lossfn(z,y);loss.backward();opt.step();total+=loss.item()*len(x)
        m.eval();ys=[];ps=[]
        with torch.no_grad():
            for x,y in va_dl:ys.append(y.numpy());ps.append(torch.sigmoid(m(x)).numpy())
        yt=np.concatenate(ys);pp=np.concatenate(ps);pred=(pp>=.5).astype(int);macro=float(f1_score(yt,pred,average='macro',zero_division=0));micro=float(f1_score(yt,pred,average='micro',zero_division=0));history.append({'epoch':ep,'macro_f1':macro,'micro_f1':micro,'train_loss':total/len(tr)});print(history[-1])
        if macro>best:best=macro;torch.save({'state_dict':m.state_dict(),'labels':LABELS,'image_size':SIZE,'threshold':.5,'architecture':'SmallRetinaCNN-v1'},MODEL_DIR/'retina_multilabel_cnn.pt')
    json.dump({'best_macro_f1':best,'history':history,'train_count':len(tr),'val_count':len(va)},open(MODEL_DIR/'training_metrics_retrained.json','w'),indent=2)
if __name__=='__main__':main()
