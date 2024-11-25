library(ggplot2)
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(RColorBrewer)

cols1 <- brewer.pal(5, "Set1")

patient_info <- read_tsv("patient_info.txt")
patient_info$studynumber <- paste(patient_info$patient_id,tolower(patient_info$LR),sep="_")
patient_info$disease <- "RP"

control_patient_info <- read_tsv("control_patient_info.txt")
control_patient_info$studynumber <- paste(control_patient_info$patient_id,tolower(control_patient_info$LR),sep="_")
control_patient_info$disease <- "nonRP"

patient_info_merged <- rbind(patient_info%>%select(c("patient_id","LR","age","sex","disease","studynumber")),control_patient_info%>%select(c("patient_id","LR","age","sex","disease","studynumber")))

shap_data <- read_csv("shap_scores.csv")
sample_id_list <- shap_data$ID %>% str_split(pattern="_",simplify = TRUE) %>% as.data.frame()
shap_data$studynumber <- paste(sample_id_list$V1,tolower(sample_id_list$V4),sep="_")

shap_mean <- shap_data %>% select(where(is.numeric)) %>% abs() %>% summarise(across(everything(), mean)) %>% t() %>% as.data.frame()
top_features <- rownames(shap_mean[head(rev(order(shap_mean$V1)),10),,drop=F])
shap_mean_top <- shap_mean[top_features,,drop=F]
shap_mean_top$feature <- factor(rownames(shap_mean_top),levels=rev(rownames(shap_mean_top)))

data_merged <- left_join(patient_info_merged,shap_data%>%select(all_of(c("studynumber",top_features))),by="studynumber")
data_merged_long <- data_merged %>% pivot_longer(cols = -c("patient_id","LR","age","sex","disease","studynumber"), names_to = "feature", values_to = "shap_score")
data_merged_long$feature <- factor(data_merged_long$feature,levels=rev(rownames(shap_mean_top)))

p<- ggplot(data_merged_long, aes(x=feature, y=shap_score, group=disease, colour=disease)) +
	geom_point(position=position_jitter(width=0.15),alpha=0.7) +
	coord_flip() +
	scale_colour_brewer(palette="Set1")+
	theme_classic(base_size=16) +
	theme(aspect.ratio = 2)
ggsave(file = "RP_diagnosis_shap_score.pdf", plot = p,width=6,height=4)


p<- ggplot(shap_mean_top, aes(x=feature, y=V1)) +
	geom_bar(fill=cols1[2],colour=NA,stat = "identity") +
	coord_flip() +
	theme_classic(base_size=16) +
	theme(aspect.ratio = 2)
ggsave(file = "RP_diagnosis_shap_score_bar.pdf", plot = p,width=6,height=4)
