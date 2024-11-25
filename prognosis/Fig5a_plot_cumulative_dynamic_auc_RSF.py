from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


n_rep = 10

var_imp = pd.read_csv("../feature1792_DaysToOutcome>0/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt",sep="\t")
var_imp["mean"] = var_imp.iloc[:,1:].mean(axis=1)
top_20_vars = list(var_imp.sort_values("mean", ascending=False).iloc[:10,:]["Unnamed: 0"])

imp_vars = list(var_imp.sort_values('mean', ascending=False).iloc[0:5,0])

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

dat = pd.merge(feature.loc[:,["studynumber"]+top_20_vars],patient_info.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome","sex"]],on="studynumber").drop("studynumber",axis=1).dropna()
dat = dat.loc[dat["DaysToOutcome"] > 0,:]

train_list = [i for i in range(0,len(dat.index)) if i % 3 != 2]
test_list = [i for i in range(0,len(dat.index)) if i % 3 == 2]

y = dat.loc[:,["eyesightloss<0.3","DaysToOutcome","sex"]]
y["eyesightloss<0.3"] = [True if i==1 else False for i in y["eyesightloss<0.3"]]

y_train = y.iloc[train_list,:].drop("sex",axis=1)

y_test = y.iloc[test_list,:]
y_test_M = y_test.loc[y_test["sex"]=="M",:].drop("sex",axis=1)
y_test_F = y_test.loc[y_test["sex"]=="F",:].drop("sex",axis=1)
y_test = y_test.drop("sex",axis=1)

yy_train = y_train.to_records(index=False)

yy_test = y_test.to_records(index=False)
yy_test_M = y_test_M.to_records(index=False)
yy_test_F = y_test_F.to_records(index=False)

X = dat.drop(["eyesightloss<0.3","DaysToOutcome"],axis=1)

X_train = X.iloc[train_list,:]
X_test = X.iloc[test_list,:]
X_test_M = X_test.loc[X_test["sex"]=="M",:].drop("sex",axis=1)
X_test_F = X_test.loc[X_test["sex"]=="F",:].drop("sex",axis=1)
X_test = X_test.drop("sex",axis=1)

times = np.arange(300,3000,100)
print(times)

rsf_auc_summary = pd.DataFrame([])
rsf_auc_summary_M = pd.DataFrame([])
rsf_auc_summary_F = pd.DataFrame([])
# rf_auc_summary = []
rsf_mean_auc_summary = []
rsf_mean_auc_summary_F = []
rsf_mean_auc_summary_M = []


for i in range(10):
	with open("./RSF_models/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep"+str(i+1)+".sav", 'rb') as f:
		rsf = cPickle.load(f)

	rsf_chf_funcs = rsf.predict_cumulative_hazard_function(
    X_test, return_array=False)
	rsf_risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])

	rsf_chf_funcs_M = rsf.predict_cumulative_hazard_function(
    X_test_M, return_array=False)
	rsf_risk_scores_M = np.row_stack([chf(times) for chf in rsf_chf_funcs_M])

	rsf_chf_funcs_F = rsf.predict_cumulative_hazard_function(
    X_test_F, return_array=False)
	rsf_risk_scores_F = np.row_stack([chf(times) for chf in rsf_chf_funcs_F])

	rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
	    yy_train, yy_test, rsf_risk_scores, times
	)

	rsf_auc_M, rsf_mean_auc_M = cumulative_dynamic_auc(
	    yy_train, yy_test_M, rsf_risk_scores_M, times
	)

	rsf_auc_F, rsf_mean_auc_F = cumulative_dynamic_auc(
	    yy_train, yy_test_F, rsf_risk_scores_F, times
	)

	rsf_auc_summary = pd.concat([rsf_auc_summary, pd.Series(rsf_auc)], axis=1)
	rsf_auc_summary_M = pd.concat([rsf_auc_summary_M, pd.Series(rsf_auc_M)], axis=1)
	rsf_auc_summary_F = pd.concat([rsf_auc_summary_F, pd.Series(rsf_auc_F)], axis=1)
	rsf_mean_auc_summary.append(rsf_mean_auc)
	rsf_mean_auc_summary_M.append(rsf_mean_auc_M)
	rsf_mean_auc_summary_F.append(rsf_mean_auc_F)


# rsf_auc_ave = rsf_auc_summary.mean(axis=1)
rsf_mean_auc_ave = np.nanmean(rsf_mean_auc_summary)
rsf_mean_auc_ave_M = np.nanmean(rsf_mean_auc_summary_M)
rsf_mean_auc_ave_F = np.nanmean(rsf_mean_auc_summary_F)

print(rsf_mean_auc_ave)
print(rsf_mean_auc_ave_M)
print(rsf_mean_auc_ave_F)

fig = plt.figure()

plt.plot(times, rsf_auc, marker="o", color="black")
plt.plot(times, rsf_auc_M, marker="o", color="blue")
plt.plot(times, rsf_auc_F, marker="o", color="red")
plt.axhline(rsf_mean_auc_ave, linestyle="--", color="black")
plt.axhline(rsf_mean_auc_ave_M, linestyle="--", color="blue")
plt.axhline(rsf_mean_auc_ave_F, linestyle="--", color="red")
plt.xlabel("days after the examination")
plt.ylabel("time-dependent AUC")
plt.grid(True)

fig.savefig("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_cumulative_dynamic_auc.pdf")


