import pandas as pd
import glob

# l0ad surv shaps
df_raw = pd.read_table('./survshap/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep1_survSHAP_aggregated.txt') 

df_list = df_raw.iloc[:,:3]
df_score_sum = df_raw.iloc[:,3:]
df_score_sum[:] = 0

# calculate surv shap average 
for i in range(10):
    i = i + 1
    print(i)
    df = pd.read_table('./survshap/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_rep'+str(i)+'_survSHAP_aggregated.txt') 
    df = df.iloc[:,3:]
    df_score_sum = df_score_sum + df
df_score_mean = df_score_sum/10

df_merged = pd.concat([df_list,df_score_mean],axis = 1)

# Save the average 
df_merged.to_csv("./scripts/RP_eyesight_loss_0.3_RSF_DaysToOutcome_shap_value_mean.csv",index = False)

columns = df_merged.columns
col_list = col_list =[columns[0]]+columns[3:].tolist()

# Corresponding image path and surv shap
pictures_list = pd.DataFrame(index = [], columns = col_list)

for i in range(len(df_merged)):
    name = df_merged.iat[i,0].split("_")
    list = glob.glob("./circle_detection/images/"+name[0]+"_*"+name[1]+"_*")

    picture_list = pd.DataFrame(index = [], columns = col_list)
    picture_list['studynumber']= list
    picture_list.iloc[:,1:] = df_merged.iloc[i,3:]

    pictures_list = pd.concat([pictures_list, picture_list])

for i in range(len(pictures_list)):
    pictures_list.iat[i,0] = pictures_list .iat[i,0][26:-4]

#Save 
pictures_list.to_csv("./scripts/surv_shap_mean.csv",index = False)


