from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import os, random


n_rep = 10

feature = pd.read_csv("../特徴量_1792.csv",header=0)

studynumber = []
for i in range(len(feature)):
    if feature['ID'][i].split("_")[3] in ["r","l"]:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[3])
    else:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[2])
feature["studynumber"] = studynumber
patient_info = pd.read_csv("../patient_info.txt",sep="\t",header=0)
patient_info["studynumber"] = patient_info["patient_id"].astype(str).str.cat(patient_info["LR"].str.lower(),sep="_").astype(str)

dat = pd.merge(feature.loc[:,["studynumber"]],patient_info.loc[:,["patient_id","studynumber","eyesightloss<0.3","DaysToOutcome","LR","age","sex","cataract","cataract_surgery"]],on="studynumber").drop("studynumber",axis=1).dropna()
dat = dat.loc[dat["DaysToOutcome"] > 0,:]

patient_list = list(set(dat["patient_id"]))
train_list = [patient_list[i] for i in range(0,len(patient_list)) if i % 3 != 2]
test_list = [patient_list[i] for i in range(0,len(patient_list)) if i % 3 == 2]

dat["sex"] = [1 if x == "M" else 0 for x in dat["sex"]]
dat["LR"] = [1 if x == "L" else 0 for x in dat["LR"]]

dat_train = dat.loc[dat["patient_id"].isin(train_list),:]
dat_test = dat.loc[dat["patient_id"].isin(test_list),:]

y_train = dat_train.loc[:,["eyesightloss<0.3","DaysToOutcome"]]
y_train["eyesightloss<0.3"] = [True if i==1 else False for i in y_train["eyesightloss<0.3"]]
yy_train = y_train.to_records(index=False)
X_train = dat_train.drop(["patient_id","eyesightloss<0.3","DaysToOutcome"],axis=1)

y_test = dat_test.loc[:,["eyesightloss<0.3","DaysToOutcome"]]
y_test["eyesightloss<0.3"] = [True if i==1 else False for i in y_test["eyesightloss<0.3"]]
yy_test = y_test.to_records(index=False)
X_test = dat_test.drop(["patient_id","eyesightloss<0.3","DaysToOutcome"],axis=1)

imp_summary = pd.DataFrame([])
cindex_train = []
cindex_test = []

os.makedirs("./CPH_models", exist_ok=True)
for i in range(n_rep):

    cph = CoxPHSurvivalAnalysis()
    cph.fit(X_train, yy_train)
    with open("./CPH_models/RP_eyesight_loss<0.3_CPH_DaysToOutcome>0_BGalone_rep"+str(i+1)+".sav", 'wb') as f:
        cPickle.dump(cph, f)

    cindex_train.append(cph.score(X_train, yy_train))
    cindex_test.append(cph.score(X_test, yy_test))

    coef = pd.DataFrame(cph.coef_).mean(axis=1)

    imp_summary = pd.concat([imp_summary, pd.DataFrame(coef)], axis=1)

cindex_summary = pd.DataFrame([])
cindex_summary["rep"] = list(range(10))
cindex_summary["train"] = cindex_train
cindex_summary["test"] = cindex_test

cindex_summary.to_csv("RP_eyesight_loss<0.3_CPH_DaysToOutcome>0_BGalone_cindex.txt",sep="\t",index=False)

imp_summary.columns = [ "rep_"+str(i+1) for i in range(n_rep)]
imp_summary.index = X_train.columns
imp_summary["mean"] = imp_summary.mean(axis=1)

imp_summary.to_csv("RP_eyesight_loss<0.3_CPH_DaysToOutcome>0_BGalone_varimp.txt",sep="\t")


