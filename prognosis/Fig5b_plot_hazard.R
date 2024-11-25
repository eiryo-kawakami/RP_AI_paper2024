library(ggplot2)
library(readr)
library(dplyr)
library(ggsci)


varimp <- read_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_varimp.txt")
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

group <- c()
for (i in 1:nrow(patient_info)){
	if(patient_info$`eyesightloss<0.3`[i]==0){
		group <- c(group,"no_loss")
	} else if(patient_info$DaysToOutcome[i] == 0){
		group <- c(group,"already")
	} else if(patient_info$DaysToOutcome[i] < 365){
		group <- c(group,"within_1year")
	} else if(patient_info$DaysToOutcome[i] < 1095){
		group <- c(group,"within_3years")
	} else if(patient_info$DaysToOutcome[i] < 1825){
		group <- c(group,"within_5years")
	} else {
		group <- c(group,"over_5years")
	}
}

patient_info$group <- factor(group,levels=c("already","within_1year","within_3years","within_5years","over_5years","no_loss"))

surv_func <- read_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_test+already_chf_rep1.txt")

chf_1year <- 1 - surv_func %>% pull("364.0")
chf_2year <- 1 - surv_func %>% pull("729.0")
chf_3year <- 1 - surv_func %>% pull("1040.0")
chf_4year <- 1 - surv_func %>% pull("1436.0")
chf_5year <- 1 - surv_func %>% pull("1827.0")

for (i in 1:10){

	surv_func <- read_tsv(paste0("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_test+already_chf_rep",i,".txt"))

	chf_1year <- chf_1year + 1 - surv_func %>% pull("364.0")
	chf_2year <- chf_2year + 1 - surv_func %>% pull("729.0")
	chf_3year <- chf_3year + 1 - surv_func %>% pull("1040.0")
	chf_4year <- chf_4year + 1 - surv_func %>% pull("1436.0")
	chf_5year <- chf_5year + 1 - surv_func %>% pull("1827.0")
}

chf_1year <- chf_1year / 10
chf_2year <- chf_2year / 10
chf_3year <- chf_3year / 10
chf_4year <- chf_4year / 10
chf_5year <- chf_5year / 10

chf_data <- data.frame(studynumber=surv_func["studynumber"],chf_1year=chf_1year,chf_2year=chf_2year,chf_3year=chf_3year,chf_4year=chf_4year,chf_5year=chf_5year)
chf_data <- left_join(chf_data,patient_info,by="studynumber")

chf_data %>% write_tsv("RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_top10vars_test+already_chf_summary.txt")

ggplot(data=chf_data,aes(x=group,y=chf_1year)) +
	stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot(aes(fill=group),width = 0.6,outlier.shape = NA)+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_1year_by_group.pdf"),width=5,height=3)

ggplot(data=chf_data,aes(x=group,y=chf_2year)) +
	stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot(aes(fill=group),width = 0.6,outlier.shape = NA)+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_2year_by_group.pdf"),width=5,height=3)

ggplot(data=chf_data,aes(x=group,y=chf_3year)) +
	stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot(aes(fill=group),width = 0.6,outlier.shape = NA)+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_3year_by_group.pdf"),width=5,height=3)

ggplot(data=chf_data,aes(x=group,y=chf_4year)) +
	stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot(aes(fill=group),width = 0.6,outlier.shape = NA)+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_4year_by_group.pdf"),width=5,height=3)

ggplot(data=chf_data,aes(x=group,y=chf_5year)) +
	stat_boxplot(geom ='errorbar', width = 0.6)+
	geom_boxplot(aes(fill=group),width = 0.6,outlier.shape = NA)+
	# geom_jitter(aes(colour=group),size=1,position = position_jitter(0.2))+
	# scale_y_continuous(limits = quantile(tmp_data$value, c(0.1, 0.9)))+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file=paste0("CHF_5year_by_group.pdf"),width=5,height=3)