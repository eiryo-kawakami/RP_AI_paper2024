from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
import _pickle as cPickle
# import eli5
# from eli5.sklearn import PermutationImportance

n_rep = 10

feature = pd.read_csv("../特徴量_1792.csv",header=0)

var_imp = pd.read_csv("../feature1792_DaysToOutcome>0/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt",sep="\t")
var_imp["mean"] = var_imp.iloc[:,1:].mean(axis=1)
top_vars = list(var_imp.sort_values("mean", ascending=False).iloc[:10,:]["Unnamed: 0"])

studynumber = []
for i in range(len(feature)):
    if feature['ID'][i].split("_")[3] in ["r","l"]:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[3])
    else:
        studynumber.append(feature['ID'][i].split("_")[0]+"_"+feature['ID'][i].split("_")[2])
feature["studynumber"] = studynumber
patient_info = pd.read_csv("../patient_info.txt",sep="\t",header=0)
patient_info["studynumber"] = patient_info["patient_id"].astype(str).str.cat(patient_info["LR"].str.lower(),sep="_").astype(str)

dat = pd.merge(feature.loc[:,["studynumber"]+top_vars],patient_info.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]],on="studynumber")

dat_already = dat.loc[dat["DaysToOutcome"] == 0,:]
dat = dat.loc[dat["DaysToOutcome"] > 0,:]


train_list = [i for i in range(0,len(dat.index)) if i % 3 != 2]
test_list = [i for i in range(0,len(dat.index)) if i % 3 == 2]

dat_test = dat.iloc[test_list,:]

dat_merged = pd.concat([dat_already,dat_test])

X = dat_merged.drop(["studynumber","eyesightloss<0.3","DaysToOutcome"],axis=1)


for i in range(10):
	with open("./RSF_models/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep"+str(i+1)+".sav", 'rb') as f:
		rsf = cPickle.load(f)

	# train_chf = pd.DataFrame(rsf.predict_survival_function(X,return_array=True))
	# print(train_chf)
	# train_chf.columns = rsf.unique_times_
	# train_chf = pd.concat([dat_train.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]].reset_index(drop=True),train_chf],axis=1)

	# train_chf.to_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top20vars_train_chf_rep"+str(i+1)+".txt",sep="\t",index=False)

	test_chf = pd.DataFrame(rsf.predict_survival_function(X,return_array=True))
	test_chf.columns = rsf.unique_times_
	test_chf = pd.concat([dat_merged.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]].reset_index(drop=True),test_chf],axis=1)

	test_chf.to_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_test+already_chf_rep"+str(i+1)+".txt",sep="\t",index=False)
