library(ggplot2)
library(readr)
library(dplyr)
library(ggsci)


df <- read_tsv("../patient_info.txt")
df <- df %>% filter(DaysToOutcome > 0)
df$`eyesightloss<0.3` <- factor(df$`eyesightloss<0.3`,levels=c("1","0"))

g <- ggplot(df, aes(x = DaysToOutcome, fill = `eyesightloss<0.3`), color="black")
g <- g + geom_histogram(position = "identity", alpha = 0.8)
g <- g + xlim(0,3000)
g <- g + scale_fill_npg()
g <- g + theme_classic(base_size=12)

ggsave(plot=g, file="DaysToOutcome.pdf",width=5,height=1.5)
