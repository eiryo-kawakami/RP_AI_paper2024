
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np

import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

type = "ResNet152"

classifier = "classifier."+type+".inference.simple"

for i in range(5):
    number = str(i)
    name = "classifier."+type+".cv_"+number+".inference.simple"
    path = "./preds/"+name+".csv"
    df = pd.read_csv(path)

    df_processed = df.filter(items=['ID','RP'])
    df_processed = df_processed.sort_values('ID')

    df_answer = pd.read_csv("./storage/data_Labels.csv")
    df_answer_processed = df_answer.filter(items=['ID','RP'])
    df_answer_processed = df_answer_processed.sort_values('ID')
    for idx in range(len(df_answer_processed)):
        df_answer_processed.iat[idx,0] = df_answer_processed.iat[idx,0][:-4]
    df_probability = pd.merge(df_answer_processed,df_processed, how='left',on =  "ID")
    df_probability.columns = ["ID","answer","probability"]
    df_probability.to_csv("auc for each classifier/" +type+ '/'+"auc_"+name+".csv",index=False)

    df_list = df_processed["RP"].values.tolist()
    df_answer_list = df_answer_processed["RP"].values.tolist()
    roc = roc_curve(df_answer_list,df_list)
    fpr, tpr, thresholds = roc
    print(roc_auc_score(df_answer_list,df_list))
    plt.plot(fpr, tpr, marker=',',label="cv_"+str(i))

plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0)
plt.grid()
plt.savefig("auc for each classifier/" +type+ '/'+'ROC_'+classifier+'.pdf')
plt.close()

for i in range(5):
    number = str(i)
    name = "classifier."+type+".cv_"+number+".inference.simple"
    path = "./preds/"+name+".csv"
    df = pd.read_csv(path)

    df_processed = df.filter(items=['ID','RP'])
    df_processed = df_processed.sort_values('ID')
    df_answer = pd.read_csv("./storage/data_Labels.csv")
    df_answer_processed = df_answer.filter(items=['ID','RP'])
    df_answer_processed = df_answer_processed.sort_values('ID')
    for idx in range(len(df_answer_processed)):
        df_answer_processed.iat[idx,0] = df_answer_processed.iat[idx,0][:-4]
    df_probability = pd.merge(df_answer_processed,df_processed, how='left',on =  "ID")
    df_probability.columns = ["ID","answer","probability"]
    df_probability.to_csv("auc for each classifier/" +type+ '/'+"auc_"+name+".csv",index=False)

    df_list = df_processed["RP"].values.tolist()
    df_answer_list = df_answer_processed["RP"].values.tolist()
    roc = precision_recall_curve(df_answer_list,df_list)
    presicion, recall, thresholds = roc
    print("cv_"+str(i) +":"+ str(average_precision_score(df_answer_list,df_list,average="micro")))
    plt.plot(recall, presicion, marker='.',markersize=1,label="cv_"+str(i))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0)
plt.grid()
plt.savefig("auc for each classifier/" +type+ '/'+'presicion_recall_'+name+'.pdf')
plt.close()



classifier = "classifier."+type+".inference.augmenting"
for i in range(5):
    number = str(i)
    name = "classifier."+type+".cv_"+number+".inference.augmenting"
    path = "./preds/"+name+".csv"
    df = pd.read_csv(path)

    df_processed = df.filter(items=['ID','RP'])
    df_processed = df_processed.sort_values('ID')

    df_answer = pd.read_csv("./storage/data_Labels.csv")
    df_answer_processed = df_answer.filter(items=['ID','RP'])
    df_answer_processed = df_answer_processed.sort_values('ID')
    for idx in range(len(df_answer_processed)):
        df_answer_processed.iat[idx,0] = df_answer_processed.iat[idx,0][:-4]
    df_probability = pd.merge(df_answer_processed,df_processed, how='left',on =  "ID")
    df_probability.columns = ["ID","answer","probability"]
    df_probability.to_csv("auc for each classifier/" +type+ '/'+"auc_"+name+".csv",index=False)

    df_list = df_processed["RP"].values.tolist()
    df_answer_list = df_answer_processed["RP"].values.tolist()
    roc = roc_curve(df_answer_list,df_list)
    fpr, tpr, thresholds = roc
    print(roc_auc_score(df_answer_list,df_list))
    plt.plot(fpr, tpr, marker=',',label="cv_"+str(i))

plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0)
plt.grid()
plt.savefig("auc for each classifier/" +type+ '/'+'ROC_'+classifier+'.pdf')
plt.close()

for i in range(5):
    number = str(i)
    name = "classifier."+type+".cv_"+number+".inference.augmenting"
    path = "./preds/"+name+".csv"
    df = pd.read_csv(path)

    df_processed = df.filter(items=['ID','RP'])
    df_processed = df_processed.sort_values('ID')
    df_answer = pd.read_csv("./storage/data_Labels.csv")
    df_answer_processed = df_answer.filter(items=['ID','RP'])
    df_answer_processed = df_answer_processed.sort_values('ID')
    for idx in range(len(df_answer_processed)):
        df_answer_processed.iat[idx,0] = df_answer_processed.iat[idx,0][:-4]
    df_probability = pd.merge(df_answer_processed,df_processed, how='left',on =  "ID")
    df_probability.columns = ["ID","answer","probability"]
    df_probability.to_csv("auc for each classifier/" +type+ '/'+"auc_"+name+".csv",index=False)

    df_list = df_processed["RP"].values.tolist()
    df_answer_list = df_answer_processed["RP"].values.tolist()
    roc = precision_recall_curve(df_answer_list,df_list)
    presicion, recall, thresholds = roc
    print("cv_"+str(i) +":"+ str(average_precision_score(df_answer_list,df_list,average="micro")))
    plt.plot(recall, presicion, marker='.',markersize=1,label="cv_"+str(i))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0)
plt.grid()
plt.savefig("auc for each classifier/" +type+ '/'+'presicion_recall_'+name+'.pdf')
plt.close()