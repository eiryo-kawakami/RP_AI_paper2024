library(ggplot2)
library(readr)
library(dplyr)
library(RColorBrewer)
library(ggsci)

cols1 <- brewer.pal(5, "Set1")

varimp <- read_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_block_split_varimp.txt")
varimp <- varimp[rev(order(rowMeans(varimp[,2:ncol(varimp)]))),]
imp_vars <- head(varimp$`...1`,10)

feature_values <- read_csv("../特徴量_1792.csv")

studynumber <- c()
for (i in 1:nrow(feature_values)){
    if (strsplit(feature_values$ID[i],split="_")[[1]][4] %in% c("r","l")){
        studynumber <- c(studynumber,paste(strsplit(feature_values$ID[i],split="_")[[1]][1],strsplit(feature_values$ID[i],split="_")[[1]][4],sep="_"))
    }
    else {
        studynumber <- c(studynumber,paste(strsplit(feature_values$ID[i],split="_")[[1]][1],strsplit(feature_values$ID[i],split="_")[[1]][3],sep="_"))
    }
}

feature_values$studynumber <- studynumber

patient_info <- read_tsv("../patient_info.txt")
patient_info$studynumber <- paste(patient_info$patient_id,tolower(patient_info$LR),sep="_")

group_700 <- c()
for (i in 1:nrow(patient_info)){
	if(patient_info$`eyesightloss<0.3`[i]==0){
		group_700 <- c(group_700,"no_loss")
	} else if(patient_info$DaysToOutcome[i] == 0){
		group_700 <- c(group_700,"already")
	} else if(patient_info$DaysToOutcome[i] < 700){
		group_700 <- c(group_700,"within_700days")
	} else {
		group_700 <- c(group_700,"over_700days")
	}
}

group_1400 <- c()
for (i in 1:nrow(patient_info)){
	if(patient_info$`eyesightloss<0.3`[i]==0){
		group_1400 <- c(group_1400,"no_loss")
	} else if(patient_info$DaysToOutcome[i] == 0){
		group_1400 <- c(group_1400,"already")
	} else if(patient_info$DaysToOutcome[i] < 1400){
		group_1400 <- c(group_1400,"within_1400days")
	} else {
		group_1400 <- c(group_1400,"over_1400days")
	}
}

patient_info$group_700 <- factor(group_700,levels=c("within_700days","over_700days","no_loss"))
patient_info$group_1400 <- factor(group_1400,levels=c("within_1400days","over_1400days","no_loss"))

surv_func <- read_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_block_split_test+already_chf_rep1.txt")

chf_700days <- 1 - surv_func %>% pull("729.0")
chf_1400days <- 1 - surv_func %>% pull("1436.0")

for (i in 1:10){

	surv_func <- read_tsv(paste0("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_block_split_test+already_chf_rep",i,".txt"))

	chf_700days <- chf_700days + 1 - surv_func %>% pull("729.0")
	chf_1400days <- chf_1400days + 1 - surv_func %>% pull("1436.0")

}

chf_700days <- chf_700days / 10
chf_1400days <- chf_1400days / 10

chf_data <- data.frame(studynumber=surv_func["studynumber"],chf_700days=chf_700days,chf_1400days=chf_1400days)
chf_data <- left_join(chf_data,patient_info,by="studynumber") %>% na.omit()

chf_data %>% write_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_block_split_test+already_chf_700days.txt")

ggplot(data=chf_data,aes(x=group_700,y=chf_700days,colour=sex)) +
	# stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot()+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_colour_npg()+
	# scale_colour_manual(values=c(cols1[c(1,5,2,3)],"grey40"))+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_block_split_700days_by_group.pdf"),width=5,height=2.5)

ggplot(data=chf_data,aes(x=group_1400,y=chf_1400days,colour=sex)) +
	# stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot()+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_colour_npg()+
	# scale_colour_manual(values=c(cols1[c(1,5,2,3)],"grey40"))+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_block_split_1400days_by_group.pdf"),width=5,height=2.5)
