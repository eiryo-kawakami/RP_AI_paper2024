library(ggplot2)
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(RColorBrewer)
# library(minerva)

cols1 <- brewer.pal(5, "Set1")

patient_info <- read_tsv("patient_info_logMAR.txt")
patient_info$studynumber <- paste(patient_info$patient_id,tolower(patient_info$LR),sep="_")
patient_info$disease <- "RP"

shap_data <- read_csv("shap_scores.csv")
sample_id_list <- shap_data$ID %>% str_split(pattern="_",simplify = TRUE) %>% as.data.frame()
shap_data$studynumber <- paste(sample_id_list$V1,tolower(sample_id_list$V4),sep="_")

shap_mean <- shap_data %>% select(where(is.numeric)) %>% abs() %>% summarise(across(everything(), mean)) %>% t() %>% as.data.frame()
top_features <- rownames(shap_mean[head(rev(order(shap_mean$V1)),10),,drop=F])
shap_mean_top <- shap_mean[top_features,,drop=F]
shap_mean_top$feature <- factor(rownames(shap_mean_top),levels=rev(rownames(shap_mean_top)))

feature_df <- read_csv("特徴量_1792.csv")
feature_df <- feature_df %>% select(all_of(c("ID",top_features)))
sample_id_list <- feature_df$ID %>% str_split(pattern="_",simplify = TRUE) %>% as.data.frame()
feature_df$studynumber <- paste(sample_id_list$V1,tolower(sample_id_list$V4),sep="_")
feature_df_merged <- left_join(patient_info,feature_df,by="studynumber")

res_summary <- c()
res_summary_M <- c()
res_summary_F <- c()

for (f in top_features){
	dat_tmp <- feature_df_merged %>% select(all_of(c("logMAR視力",f,"sex"))) %>% na.omit()
	colnames(dat_tmp) <- c("logMAR_eyesight","value","sex")
	dat_tmp_M <- dat_tmp %>% filter(sex=="M")
	dat_tmp_F <- dat_tmp %>% filter(sex=="F")

	res <- lm(value ~ logMAR_eyesight, data=dat_tmp)
	res2 <- data.frame(coef(summary(res)))
	res2$variable <- rownames(res2)
	res2$feature <- f
	res2$sex <- "all"

	res_M <- lm(value ~ logMAR_eyesight, data=dat_tmp_M)
	res_F <- lm(value ~ logMAR_eyesight, data=dat_tmp_F)
	res2_M <- data.frame(coef(summary(res_M)))
	res2_M$variable <- rownames(res2_M)
	res2_M$feature <- f
	res2_M$sex <- "M"

	res2_F <- data.frame(coef(summary(res_F)))
	res2_F$variable <- rownames(res2_F)
	res2_F$feature <- f
	res2_F$sex <- "F"

	res_summary <- rbind(res_summary,res2[c("logMAR_eyesight"),c("feature","variable","sex","t.value","Pr...t..")])
	res_summary_M <- rbind(res_summary_M,res2_M[c("logMAR_eyesight"),c("feature","variable","sex","t.value","Pr...t..")])
	res_summary_F <- rbind(res_summary_F,res2_F[c("logMAR_eyesight"),c("feature","variable","sex","t.value","Pr...t..")])
}

colnames(res_summary) <- c("feature","variable","sex","t_value","p_value")
res_summary$FDR <- p.adjust(res_summary$p_value)

colnames(res_summary_M) <- c("feature","variable","sex","t_value","p_value")
res_summary_M$FDR <- p.adjust(res_summary_M$p_value)

colnames(res_summary_F) <- c("feature","variable","sex","t_value","p_value")
res_summary_F$FDR <- p.adjust(res_summary_F$p_value)

res_summary_merged <- rbind(res_summary,res_summary_M,res_summary_F)

res_summary_merged$feature <- factor(res_summary_merged$feature,levels=top_features)
res_summary_merged$sex <- factor(res_summary_merged$sex,levels=c("M","F","all"))

res_summary_merged %>% write_tsv(file="feature_association_logMAR_eyesight.txt")

p <- ggplot(res_summary_merged,aes(x=feature,y=sex))+
	geom_tile(aes(fill = t_value),colour="white")+
	scale_fill_gradient2(low="blue",high="red",limit = c(-5,5),guide="colorbar",oob=scales::squish)+
	coord_fixed(ratio=1)
ggsave(file = "feature_association_logMAR_eyesight_heatmap.pdf",plot=p)
