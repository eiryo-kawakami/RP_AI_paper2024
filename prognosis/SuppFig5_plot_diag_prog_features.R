library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggsci)
library(ggrepel)
library(RColorBrewer)


myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
cols1 <- brewer.pal(5, "Set1")

diag_imp <- read_csv("../shap_scores.csv")
diag_imp_longer <- diag_imp %>% pivot_longer(cols = c(-ID), names_to = "feature", values_to = "imp")
diag_imp_mean <- diag_imp_longer %>% group_by(feature) %>% summarise(diag_imp=mean(abs(imp)))

prog_imp <- read_tsv("../feature1792_DaysToOutcome>0/RP_eyesight_loss<0.3_RSF_DaysToOutcome>0_varimp.txt")
prog_imp <- rename(prog_imp, feature = ...1)
prog_imp_longer <- prog_imp %>% pivot_longer(cols = c(-feature), names_to = "rep", values_to = "imp")
prog_imp_mean <- prog_imp_longer %>% group_by(feature) %>% summarise(prog_imp=mean(imp))

imp_mean_merged <- left_join(diag_imp_mean,prog_imp_mean,by="feature")
imp_mean_merged$prog_imp[which(imp_mean_merged$prog_imp < 0)] <- 0
imp_mean_merged$colour <- "other"
imp_mean_merged$colour[head(rev(order(imp_mean_merged$diag_imp)),10)] <- "diag"
imp_mean_merged$colour[head(rev(order(imp_mean_merged$prog_imp)),10)] <- "prog"
imp_mean_merged$colour <- factor(imp_mean_merged$colour,levels=c("diag","prog","other"))
imp_mean_merged$label <- ""
imp_mean_merged$label[head(rev(order(imp_mean_merged$diag_imp)),10)] <- imp_mean_merged$feature[head(rev(order(imp_mean_merged$diag_imp)),10)]
imp_mean_merged$label[head(rev(order(imp_mean_merged$prog_imp)),10)] <- imp_mean_merged$feature[head(rev(order(imp_mean_merged$prog_imp)),10)]


ggplot(data=imp_mean_merged,aes(x=diag_imp,y=prog_imp)) +
	geom_point(aes(colour=colour),size=2)+
	geom_text_repel(aes(label=label),force_pull=0.01,force=10,max.overlaps=100)+
	scale_colour_manual(values=c(cols1[1],cols1[2],"grey"))+
	theme_classic(base_size=12)+
	theme(axis.text.x = element_text(angle = 45, hjust = 1))+
	scale_y_continuous(limits=c(0,NA))+
    theme(aspect.ratio=1)
ggsave(file="feature1792_diagnosis_prognosis_feature_importance.pdf",width=7,height=5)
