# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:36:23 2017

@author: Ruobing
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
###paper: A preprocessing scheme for high cardinality categorical attributes in classification and prediction problems
def BayesEncoding(train,test,target,feature,k=5,f=1,noise=0.01,drop_origin_feature=False):  
    entry=pd.get_dummies(train[target])
    classes=list(entry)
    entry[feature]=train[feature]
    entry=entry.groupby(feature).agg('sum').reset_index()
    entry['total_count']=entry.sum(axis=1)
    for e in classes:
        newname='bayes_'+feature+'_'+str(e)
        prior=sum(entry[e])/sum(entry['total_count'])
        posterior=entry[e]/entry['total_count']
        B=1/(1+np.exp(-1*(entry['total_count']-k)/f))
        entry[newname]=B*posterior+(1-B)*prior

    newfeature=['bayes_'+feature+'_'+str(e) for e in classes]
    newlist=[feature]+ newfeature   
    useful=entry[newlist]
    train=train.join(useful.set_index(feature),on=feature)
    test=test.join(useful.set_index(feature),on=feature)

    for e in classes:
        newname='bayes_'+feature+'_'+str(e)
        test[newname]=test[newname].fillna(sum(entry[e])/sum(entry['total_count']))
        if noise:  # Add uniform noise. Not mentioned in original paper
            train[newname]=train[newname]*np.random.uniform(1 - noise, 1 + noise, len(train[newname]))
            test[newname]=test[newname]*np.random.uniform(1 - noise, 1 + noise, len(test[newname]))
        
    
    if drop_origin_feature==True:
        train=train.drop(feature,1)
        test=test.drop(feature,1)
        
    return train,test

def BayesEncodingKfold(train,test,target,feature,k=5,f=1,noise=0.01,drop_origin_feature=False,fold=5):
    train_no_use,test_useful=BayesEncoding(train,test,target,feature,k,f,noise,drop_origin_feature)
    skf = StratifiedKFold(fold)
    alltrain_min=[]
    for train_id, test_id in skf.split(train,np.zeros(len(train))):
        train_maj,train_min=BayesEncoding(train.iloc[train_id], train.iloc[test_id],target,feature,k,f,noise,drop_origin_feature)
        alltrain_min.append(train_min)
    train_useful=pd.concat(alltrain_min,0)
    return train_useful,test_useful
    
    
def LOOEncoding(train,test,target,feature,noise=0.01,drop_origin_feature=False):
    cs = train.groupby(by=[feature])[target].sum()
    cc = train[feature].value_counts()
    boolean = (cc == 1)
    index = boolean[boolean == True].index.values
    cc.loc[boolean] += 1
    cs.loc[index] *= 2
    train = train.join(cs.rename('sum'), on=[feature])
    train = train.join(cc.rename('count'), on=[feature])
    newname='loo_'+feature
    train[newname] = (train['sum']-train[target])/(train['count'] - 1)
    if noise: train[newname]= train[newname]*np.random.uniform(1 - noise, 1 + noise, len(train[newname]))  # Add uniform noise. Not mentioned in original paper
    del train['sum'], train['count']
    cstest=train.groupby(by=[feature])[newname].mean()
    test=test.join(cstest.rename(newname),on=[feature])
    if drop_origin_feature==True:
        train=train.drop(feature,1)
        test=test.drop(feature,1)
    return train,test

def LOOEncodingKfold(train,test,target,feature,noise=0.01,drop_origin_feature=False,fold=5):
    train_no_use,test_useful=LOOEncoding(train,test,target,feature,noise,drop_origin_feature)
    skf = StratifiedKFold(fold)
    alltrain_min=[]
    for train_id, test_id in skf.split(train,np.zeros(len(train))):
        train_maj,train_min=LOOEncoding(train.iloc[train_id], train.iloc[test_id],target,feature,noise,drop_origin_feature)
        alltrain_min.append(train_min)
    train_useful=pd.concat(alltrain_min,0)
    newname='loo_'+feature
    train_useful[newname]=train_useful[newname].fillna(train_useful[newname].median())
    test_useful[newname]=test_useful[newname].fillna(test_useful[newname].median())
    return train_useful,test_useful