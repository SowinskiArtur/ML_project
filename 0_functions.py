#stworzymy specjalna funkcje do logarytmowania zmiennej, które zamieni ujemne wartosci na zera, w przypadku kiedy wystapia

import numpy as np
import pandas as pd

def log_func(df,column,valid,features,precision,stand=False,*args,**kwargs):
    features2=features.copy()
    features2.remove(column)
    features2.append(f'{column}_log')
    df2=df.copy()
    df2[f'{column}_log']=df2[column].copy()
    #w przypadku wystepowania zmiennych ujemnych zamieniamy na zero
    if len(df2.loc[df2[f'{column}_log']<0])!=0:
        df2.loc[df2[f'{column}_log']<0,f'{column}_log']=0
    df2[f'{column}_log']=np.log(df2[f'{column}_log']+1)
    if stand:
        df2[f'{column}_log']=df2[[f'{column}_log']].apply(lambda x: (x-np.mean(x))/np.std(x))
    prec=np.mean(valid(df=df2,features=features2,*args,**kwargs)[0])
    print(prec)
    if prec>precision:
        print('Poprawa modelu')
        df[f'{column}_log']=df2[f'{column}_log']
        features.remove(column)
        features.append(f'{column}_log')
    else:
        print('Brak poprawy modelu')

#funkcja do grupowania

def grupowanie(df,columns,new_column,valid,function,features,precision,stand=False,*args,**kwargs):
    df2=df.copy()
    df2[new_column]=df[columns].apply(function,axis=1)
    df2.drop(columns,axis=1,inplace=True)
    features2=features.copy()
    for i in columns:
        features2.remove(i)
    features2.append(new_column)
    if stand:
        df2[new_column]=df2[[new_column]].apply(lambda x: (x-np.mean(x))/np.std(x))
    prec=np.mean(valid(df=df2,features=features2,*args,**kwargs)[0])
    print(prec)
    if prec>precision:
        print('Poprawa modelu')
        df[new_column]=df2[new_column]
        for i in columns:
            features.remove(i)
        features.append(new_column)
    else:
        print('Brak poprawy modelu')

#funkcja tworzaca serie pandasa zawierającą predykcje
def create_probs_series(probs,indices):
    probs=sum(probs,[])
    indices=sum(indices,[])
    probs=pd.Series(probs,index=indices).sort_index()
    probs.index=range(1,30001)
    return probs

#funkcja do standaryzacji

def stand(features,df,valid,function=lambda x: (x-np.min(x))/(np.max(x)-np.min(x))):
    df2=df.copy()
    df2[features]=df2[features].apply(function)
    test_precisions, probs, indices=valid(df=df2)
    print(np.mean(test_precisions))
