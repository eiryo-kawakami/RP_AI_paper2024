from sksurv.metrics import cumulative_dynamic_auc,concordance_index_ipcw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


var_imp = pd.read_csv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt",sep="\t")
var_imp["mean"] = var_imp.iloc[:,1:].mean(axis=1)

imp_vars = list(var_imp.sort_values('mean', ascending=False).iloc[0:10,0])

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

dat = pd.merge(feature.iloc[:,1:],patient_info.loc[:,["patient_id","studynumber","eyesightloss<0.3","DaysToOutcome","sex"]],on="studynumber").drop("studynumber",axis=1).dropna()
dat = dat.loc[dat["DaysToOutcome"] > 0,:]

patient_list = list(set(dat["patient_id"]))
train_list = [patient_list[i] for i in range(0,len(patient_list)) if i % 3 != 2]
test_list = [patient_list[i] for i in range(0,len(patient_list)) if i % 3 == 2]

dat_train = dat.loc[dat["patient_id"].isin(train_list),:]
dat_test = dat.loc[dat["patient_id"].isin(test_list),:]

y_train = dat_train.loc[:,["eyesightloss<0.3","DaysToOutcome"]]
y_train["eyesightloss<0.3"] = [True if i==1 else False for i in y_train["eyesightloss<0.3"]]
yy_train = y_train.to_records(index=False)
X_train = dat_train.drop(["patient_id","eyesightloss<0.3","DaysToOutcome"],axis=1)

y_test = dat_test.loc[:,["eyesightloss<0.3","DaysToOutcome","sex"]]
y_test["eyesightloss<0.3"] = [True if i==1 else False for i in y_test["eyesightloss<0.3"]]
y_test_M = y_test.loc[y_test["sex"]=="M",:].drop("sex",axis=1)
y_test_F = y_test.loc[y_test["sex"]=="F",:].drop("sex",axis=1)
y_test = y_test.drop("sex",axis=1)

yy_test = y_test.to_records(index=False)
yy_test_M = y_test_M.to_records(index=False)
yy_test_F = y_test_F.to_records(index=False)

X_test = dat_test.drop(["patient_id","eyesightloss<0.3","DaysToOutcome"],axis=1)
X_test_M = X_test.loc[X_test["sex"]=="M",:].drop("sex",axis=1)
X_test_F = X_test.loc[X_test["sex"]=="F",:].drop("sex",axis=1)

X_test = X_test.drop(["sex"],axis=1)


def plot_cumulative_dynamic_auc(y_train, y_test, risk_score, label, times, color=None):
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score, times)

    plt.plot(times, auc, marker="o", color=color, label=label)
    plt.xlabel("days after the examination")
    plt.ylabel("time-dependent AUC")
    plt.axhline(mean_auc, color=color, linestyle="--")
    plt.legend()


times = np.arange(200,3000,100)
print(times)

fig = plt.figure()

for i, col in enumerate(imp_vars):
    print(i)
    print(col)
    sign = 1
    ret = concordance_index_ipcw(yy_train, yy_test, X_test.loc[:,str(col)], tau=times[-1])
    if ret[0] < 0.5:
        sign = -1
    plot_cumulative_dynamic_auc(yy_train, yy_test, sign * X_test.loc[:,str(col)], str(col), times, color="C{}".format(i))

fig.savefig("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_block_split_imp_vars_cumulative_dynamic_auc.pdf")
