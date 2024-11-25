from sksurv.metrics import cumulative_dynamic_auc,concordance_index_ipcw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


var_imp = pd.read_csv("../feature1792_DaysToOutcome>0/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt",sep="\t")
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

dat = pd.merge(feature.iloc[:,1:],patient_info.loc[:,["studynumber","eyesightloss<0.3","DaysToOutcome"]],on="studynumber").drop("studynumber",axis=1).dropna()
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

fig.savefig("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_imp_vars_cumulative_dynamic_auc.pdf")
