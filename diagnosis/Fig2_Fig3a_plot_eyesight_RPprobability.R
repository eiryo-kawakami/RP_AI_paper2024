library(ggplot2)
library(readr)
library(dplyr)
library(ggsci)
# library(ggparl)


patient_info <- read_tsv("patient_info_logMAR.txt")
patient_info$studynumber <- paste(patient_info$patient_id,tolower(patient_info$LR),sep="_")

control_patient_info <- read_tsv("control_patient_info.txt")
control_patient_info$studynumber <- paste(control_patient_info$patient_id,tolower(control_patient_info$LR),sep="_")

RPprobability <- read_csv("auc_classifier.EfficientNetB4.cv_4.inference.simple.csv")

studynumber <- c()
for (i in 1:nrow(RPprobability)){
    if (strsplit(RPprobability$ID[i],split="_")[[1]][4] %in% c("r","l")){
        studynumber <- c(studynumber,paste(strsplit(RPprobability$ID[i],split="_")[[1]][1],strsplit(RPprobability$ID[i],split="_")[[1]][4],sep="_"))
    }
    else {
        studynumber <- c(studynumber,paste(strsplit(RPprobability$ID[i],split="_")[[1]][1],strsplit(RPprobability$ID[i],split="_")[[1]][3],sep="_"))
    }
}

RPprobability$studynumber <- studynumber

RPpatient_data <- left_join(patient_info,RPprobability,by="studynumber") %>% na.omit()
RPpatient_data$cataract <- factor(RPpatient_data$cataract)
RPpatient_data$group <- "RP"

nonRPpatient_data <- left_join(control_patient_info,RPprobability,by="studynumber") %>% na.omit()
nonRPpatient_data$cataract <- "control"
nonRPpatient_data$group <- "control"

merged_data <- bind_rows(RPpatient_data%>%select(colnames(nonRPpatient_data))%>%select(-c("patient_id")),nonRPpatient_data%>%select(-c("patient_id")))

merged_data %>% write_tsv("RPprobability_eyesightstart.txt")

# res <- lm(log10(probability) ~ eyesight_start + I(eyesight_start^2), data = RPpatient_data)
# res2 <- lm(log10(probability) ~ sex, data = RPpatient_data)

ggplot(data=RPpatient_data,aes(x=`logMAR視力`,y=log10(probability))) +
	geom_point(aes(fill=LR),shape=21, size=3,alpha=0.7,color="black")+
	geom_smooth(method = "lm", formula = y ~ x, se = T)+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file="RPprobability_eyesightstart_logMAR.pdf",width=4,height=3)

ggplot(data=RPpatient_data,aes(x=`logMAR視力`,y=log10(probability))) +
	geom_point(aes(fill=sex),shape=21, size=2,alpha=0.7,color="black")+
	geom_smooth(method = "lm", formula = y ~ x, se = T)+
	scale_fill_npg()+
	facet_wrap(~sex)+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file="RPprobability_eyesightstart_logMAR_bysex.pdf",width=5,height=3)

ggplot(data=RPpatient_data,aes(x=cataract,y=log10(probability))) +
	geom_boxplot()+
	scale_fill_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file="RPprobability_cataract.pdf",width=2,height=3)

ggplot(data=merged_data,aes(x=group,y=log10(probability),color=sex)) +
	geom_boxplot()+
	scale_color_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file="RPprobability_sex.pdf",width=3,height=3)

res1 <- lm(log10(probability)~sex,data=merged_data %>% filter(group=="RP"))
res2 <- lm(log10(probability)~sex,data=merged_data %>% filter(group=="control"))

ggplot(data=merged_data,aes(x=age,y=log10(probability))) +
	geom_point(aes(color=sex,shape=group),size=3,alpha=0.7)+
	# geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = T)+
	scale_shape_manual(values=c(0,16))+
	scale_color_npg()+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file="RPprobability_age.pdf",width=4,height=3)
