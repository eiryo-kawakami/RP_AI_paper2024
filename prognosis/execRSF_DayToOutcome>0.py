from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import os

n_rep = 10

feature = pd.read_csv("../特徴量_1792.csv",header=0)

var_imp = pd.read_csv("../feature1792_DaysToOutcome>0/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt",sep="\t")
var_imp["mean"] = var_imp.iloc[:,1:].mean(axis=1)
top_10_vars = list(var_imp.sort_values("mean", ascending=False).iloc[:10,:]["Unnamed: 0"])

studynumber = []
for i in range(len(feature)):
    if feature['ID'][i].split("_")[3] in ["r","l"]:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[3])
    else:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[2])
feature["studynumber"] = studynumber
patient_info = pd.read_csv("../patient_info.txt",sep="\t",header=0)
patient_info["studynumber"] = patient_info["patient_id"].astype(str).str.cat(patient_info["LR"].str.lower(),sep="_").astype(str)

dat = pd.merge(feature.loc[:,["studynumber"]+top_10_vars],patient_info.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]],on="studynumber").drop("studynumber",axis=1).dropna()
dat = dat.loc[dat["DaysToOutcome"] > 0,:]

train_list = [i for i in range(0,len(dat.index)) if i % 3 != 2]
test_list = [i for i in range(0,len(dat.index)) if i % 3 == 2]

y = dat.loc[:,["eyesightloss<0.3","DaysToOutcome"]]
y["eyesightloss<0.3"] = [True if i==1 else False for i in y["eyesightloss<0.3"]]
yy = y.to_records(index=False)
X = dat.drop(["eyesightloss<0.3","DaysToOutcome"],axis=1)

yy_train = yy[train_list]
yy_test = yy[test_list]

X_train = X.iloc[train_list,:]
X_test = X.iloc[test_list,:]

imp_summary = pd.DataFrame([])
cindex_train = []
cindex_test = []

feature_names = X.columns.tolist()
os.makedirs("./RSF_models", exist_ok=True)
for i in range(n_rep):
    rsf = RandomSurvivalForest(n_estimators=2000,min_samples_split=10,min_samples_leaf=15,max_features="sqrt",oob_score=True,n_jobs=16,verbose=1,random_state=i)
    rsf.fit(X_train, yy_train)
    with open("./RSF_models/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep"+str(i+1)+".sav", 'wb') as f:
        cPickle.dump(rsf, f)

    cindex_train.append(rsf.oob_score_)
    cindex_test.append(rsf.score(X_test, yy_test))
    
    perm = permutation_importance(rsf, X_train, yy_train, n_repeats=10, random_state=i)
    imp_summary = pd.concat([imp_summary, pd.DataFrame(perm.importances_mean)], axis=1)

cindex_summary = pd.DataFrame([])
cindex_summary["rep"] = list(range(10))
cindex_summary["train"] = cindex_train
cindex_summary["test"] = cindex_test

cindex_summary.to_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_cindex.txt",sep="\t",index=False)

imp_summary.columns = [ "rep_"+str(i+1) for i in range(n_rep)]
imp_summary.index = X.columns

imp_summary.to_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_varimp.txt",sep="\t")

