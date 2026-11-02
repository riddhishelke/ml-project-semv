"""Experiment: does each customer's RECENT routing (strictly earlier dates only) help on future months?
Run from the project root: python src/experiment_recent_routing.py
"""
import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
df=pd.read_csv('data/processed/purlin_clean.csv',parse_dates=['DATE'])
df['M']=df.MACHINE.str.replace('MACHINE ','').str.title();CL=['New Machine','No.1','No.2']
NUM=['THICKNESS_MM','WIDTH_MM','GSM','LENGTH_MM','NO_OF_PURLIN','UNIT_WEIGHT_KG']
def recent(df,keys,days):
    out=np.zeros((len(df),4));
    for _,g in df.groupby(keys):
        idx=g.index.values;d=g.DATE.values.astype('datetime64[D]')
        order=np.argsort(d,kind='stable');idx,d=idx[order],d[order]
        oh=np.stack([(g.loc[idx,'M']==c).values.astype(float) for c in CL],1);cum=np.vstack([np.zeros(3),oh.cumsum(0)])
        hi=np.searchsorted(d,d,side='left');lo=np.searchsorted(d,d-np.timedelta64(days,'D'),side='left')
        cnt=cum[hi]-cum[lo];n=cnt.sum(1,keepdims=True)
        out[idx,:3]=np.where(n>0,cnt/np.maximum(n,1),np.nan);out[idx,3]=n[:,0]
    return out

print("== time split (train Apr-Aug, test Sep-Oct)")
for days in (14,30,60):
    R=recent(df,['PARTY_GROUP'],days);R2=recent(df,['PARTY_GROUP','THICKNESS_MM','WIDTH_MM'],days)
    X=df[NUM].copy();X['PG']=df.PARTY_GROUP.astype('category').cat.codes
    for i,c in enumerate(CL):X[f'r_{c}']=R[:,i];X[f'rs_{c}']=R2[:,i]
    X['n_r']=R[:,3];X['n_rs']=R2[:,3];Xf=X.fillna(-1)
    tr=df.DATE<'2026-09-01';te=~tr
    m=RandomForestClassifier(300,random_state=42,class_weight='balanced',min_samples_leaf=2).fit(Xf[tr],df.M[tr]);p=m.predict(Xf[te])
    print(f'lookback {days:2d}d RF: acc={accuracy_score(df.M[te],p):.3f} macroF1={f1_score(df.M[te],p,average="macro"):.3f}',flush=True)
print("== monthly retraining")
for days in (7,14):
    R=recent(df,['PARTY_GROUP'],days);R2=recent(df,['PARTY_GROUP','THICKNESS_MM','WIDTH_MM'],days)
    X=df[NUM].copy();X['PG']=df.PARTY_GROUP.astype('category').cat.codes
    for i,c in enumerate(CL):X[f'r_{c}']=R[:,i];X[f'rs_{c}']=R2[:,i]
    X['n_r']=R[:,3];X['n_rs']=R2[:,3];Xf=X.fillna(-1)
    for cut,end in [('2026-08-01','2026-09-01'),('2026-09-01','2026-10-01'),('2026-10-01','2027-01-01')]:
        tr=df.DATE<cut;te=(df.DATE>=cut)&(df.DATE<end)
        m=RandomForestClassifier(300,random_state=42,class_weight='balanced',min_samples_leaf=2).fit(Xf[tr],df.M[tr]);p=m.predict(Xf[te])
        print(f'lookback {days:2d}d predict {cut[:7]}: acc={accuracy_score(df.M[te],p):.3f} macroF1={f1_score(df.M[te],p,average="macro"):.3f} n={te.sum()}',flush=True)
