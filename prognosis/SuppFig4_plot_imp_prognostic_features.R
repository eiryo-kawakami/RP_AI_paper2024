library(ggplot2)
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(RColorBrewer)
# library(ggsci)

cols1 <- brewer.pal(5, "Set1")

patient_info <- read_tsv("../patient_info_logMAR.txt")
patient_info$studynumber <- paste(patient_info$patient_id,tolower(patient_info$LR),sep="_")
patient_info$disease <- "RP"

group <- c()
for (i in 1:nrow(patient_info)){
	if(patient_info$`eyesightloss<0.3`[i]==0){
		group <- c(group,"no_loss")
	} else if(patient_info$DaysToOutcome[i] == 0){
		group <- c(group,"already")
	} else if(patient_info$DaysToOutcome[i] < 700){
		group <- c(group,"within_700days")
	} else {
		group <- c(group,"over_700days")
	}
}

patient_info$group <- group

control_patient_info <- read_tsv("../control_patient_info.txt")
control_patient_info$studynumber <- paste(control_patient_info$patient_id,tolower(control_patient_info$LR),sep="_")
control_patient_info$disease <- "nonRP"
control_patient_info$group <- "nonRP"

patient_info_merged <- rbind(patient_info%>%select(c("patient_id","LR","age","sex","disease","studynumber","group")),control_patient_info%>%select(c("patient_id","LR","age","sex","disease","studynumber","group")))

patient_info_merged$group <- factor(patient_info_merged$group,levels=c("within_700days","over_700days","no_loss"))

varimp <- read_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_varimp.txt")
varimp <- varimp[rev(order(rowMeans(varimp[,2:ncol(varimp)]))),]
top_features <- head(varimp$`...1`,10)

feature_df <- read_csv("../特徴量_1792.csv")
feature_df <- feature_df %>% select(all_of(c("ID",top_features)))
sample_id_list <- feature_df$ID %>% str_split(pattern="_",simplify = TRUE) %>% as.data.frame()
feature_df$studynumber <- paste(sample_id_list$V1,tolower(sample_id_list$V4),sep="_")
feature_df_merged <- left_join(patient_info_merged,feature_df,by="studynumber")

feature_df_merged_long <- feature_df_merged %>% select(-c("ID")) %>% pivot_longer(cols = -c("patient_id","LR","age","sex","disease","studynumber","group"), names_to = "feature", values_to = "value") %>% na.omit()
feature_df_merged_long$feature <- factor(feature_df_merged_long$feature,levels=top_features)

ggplot(data=feature_df_merged_long,aes(x=group,y=value,color=group)) +
	geom_boxplot()+
	scale_color_manual(values=c(cols1[c(1,2,3)],"grey40"))+
	facet_grid(.~feature)+
	theme_classic(base_size=12)
ggsave(file="imp_features.pdf",width=10,height=3)


# feature_df_merged2 <- left_join(patient_info%>%select(c("patient_id","LR","age","sex","disease","studynumber","logMAR視力")),feature_df,by="studynumber")
# feature_df_merged_long2 <- feature_df_merged2 %>% select(-c("ID")) %>% pivot_longer(cols = -c("patient_id","LR","age","sex","disease","studynumber","logMAR視力"), names_to = "feature", values_to = "value")
# feature_df_merged_long2$feature <- factor(feature_df_merged_long2$feature,levels=top_features)

# feature_df_merged_long2_1 <- feature_df_merged_long2 %>% filter(feature %in% top_features[c(1:5)])
# feature_df_merged_long2_2 <- feature_df_merged_long2 %>% filter(feature %in% top_features[c(6:10)])

# p <- ggplot(data=feature_df_merged_long2_1) +
# 	geom_point(aes(x=`logMAR視力`,y=value,fill=sex),shape=21,size=1,alpha=0.7,stroke=0.5) +
# 	geom_smooth(aes(x=`logMAR視力`,y=value),method = "lm", formula = y ~ x, se = T)+
# 	scale_fill_npg() +
# 	facet_grid(sex~feature)+
# 	theme_classic(base_size = 12)+
# 	theme(aspect.ratio=1)
# ggsave(file="imp_features_eyesight_logMAR_start_1.pdf",plot = p,width=8,height=5)

# p <- ggplot(data=feature_df_merged_long2_2) +
# 	geom_point(aes(x=`logMAR視力`,y=value,fill=sex),shape=21,size=1,alpha=0.7,stroke=0.5) +
# 	geom_smooth(aes(x=`logMAR視力`,y=value),method = "lm", formula = y ~ x, se = T)+
# 	scale_fill_npg() +
# 	facet_grid(sex~feature)+
# 	theme_classic(base_size = 12)+
# 	theme(aspect.ratio=1)
# ggsave(file="imp_features_eyesight_logMAR_start_2.pdf",plot = p,width=8,height=5)

