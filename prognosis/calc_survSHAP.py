from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.model_selection import train_test_split
import os, random
from sksurv.ensemble import RandomSurvivalForest
from survshap import SurvivalModelExplainer, ModelSurvSHAP

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

dat = pd.merge(feature.loc[:,["studynumber"]+top_10_vars],patient_info.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]],on="studynumber").dropna()
dat2 = dat.copy()
dat = dat.drop("studynumber",axis=1)
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

dat_test = dat2.iloc[test_list,:]

for i in range(10):
	with open("./RSF_models/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep"+str(i+1)+".sav", 'rb') as f:
		rsf = cPickle.load(f)

	rsf_exp = SurvivalModelExplainer(rsf, X_test, yy_test)
	survshap_global_rsf = ModelSurvSHAP(random_state=i)
	survshap_global_rsf.fit(rsf_exp)

	survshap_res_summary = pd.DataFrame([])
	for j in range(len(survshap_global_rsf.individual_explanations)):
		example_rsf = survshap_global_rsf.individual_explanations[j]
		res = example_rsf.simplified_result
		survshap_res_summary = pd.concat([survshap_res_summary,res["aggregated_change"]],axis=1)

	survshap_res_summary = survshap_res_summary.T
	survshap_res_summary.columns = res.variable_name

	survshap_res_summary = pd.concat([dat_test.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]].reset_index(drop=True),survshap_res_summary.reset_index(drop=True)],axis=1)

	survshap_res_summary.to_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep"+str(i+1)+"_survSHAP_aggregated.txt",sep="\t",index=False)
