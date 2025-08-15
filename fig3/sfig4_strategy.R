library(ggplot2)
library("dplyr")
library(smplot2)
library(rstatix)
library(emmeans)

data_folder = "data"
out_folder = "figs"

## S Fig 4A Hybrid percentage

data_file = file.path(data_folder, "hybrid_count.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$condition = factor(df$condition)
df$predicted = factor(df$predicted)

hybrid_predict = ggplot(data=df, aes(x=condition  , y=calls , fill=predicted)) +
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position = position_dodge(width=0.8))+
  scale_fill_manual(values=c('#6B8A7A', '#254336'), labels=c("True","Chance"))+
  labs(y ='Epochs with 2 responses (%)', x=NULL)+theme_classic()+  facet_wrap(~experiment)
hybrid_predict

m <- lmer(calls  ~ predicted*condition+experiment +(1 | animal ) + (1|colony)  , data = df)
anova(m)

EMM <- emmeans(m, ~ predicted*condition*experiment)
test(pairs(EMM, by = c("condition", 'experiment')), by = NULL, adjust = "bh")


## Fig S4B - Stability within conditions
data_file = file.path(data_folder, "stab_within_conditions.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$condition = factor(df$condition)
df$noise_first_half = df$noise_first_half*100
df$noise_second_half = df$noise_second_half*100

stab_within_cond = ggplot(data=df, aes(x=noise_first_half  , y=noise_second_half , col=condition)) +
  geom_point()+
  labs(x ='Early response 1st half (%)', y='Early response 2nd half (%)')+
  sm_statCorr()+
  scale_color_manual(values=c('#6B8A7A', '#254336'))+
  theme_classic()+facet_wrap(~experiment)
stab_within_cond


## Fig S4C - Call response timing compared to hybrid
df_temp = df[df$epoch_type!=2,]
df_temp$epoch_type = factor(df_temp$epoch_type)
response_timing_first = ggplot(data=df_temp, aes(x=epoch_type  , y=mean_response_timing_first, fill=condition)) +
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position = position_dodge(width=0.8))+
  scale_x_discrete(breaks=c(1,3), labels=c("Onset","Both")) + 
  scale_fill_manual(values=c('#6B8A7A', '#254336'))+
  labs(y ='Response time (s)', x=NULL)+theme_classic()+  facet_wrap(~experiment)
response_timing_first


## Stats
df$epoch_factor = factor(df$epoch_type)
lmm = lmer(mean_response_timing_first  ~epoch_factor*condition +(1|animal )+ (1|colony), data =df[df$experiment=='noise',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")

lmm = lmer(mean_response_timing_first  ~epoch_factor*condition +(1|animal ), data =df[df$experiment=='sc',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")


## Fig S4D - Call response timing compared to hybrid
df_temp = df[df$epoch_type>1,]
df_temp$epoch_type = factor(df_temp$epoch_type)
response_timing_second = ggplot(data=df_temp, aes(x=epoch_type  , y=mean_response_timing_second, fill=condition)) +
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position = position_dodge(width=0.8))+
  scale_x_discrete(breaks=c(2,3), labels=c("Offset","Both")) + 
  scale_fill_manual(values=c('#6B8A7A', '#254336'))+
  labs(y ='Response time (s)', x=NULL)+theme_classic()+  facet_wrap(~experiment)
response_timing_second


#Stats
df$epoch_factor = factor(df$epoch_type)
lmm = lmer(mean_response_timing_second  ~epoch_factor*condition +(1|animal )+ (1|colony), data =df[df$experiment=='noise',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")

lmm = lmer(mean_response_timing_second  ~epoch_factor*condition +(1|animal ), data =df[df$experiment=='sc',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")



sup_plots = perc_all+hybrid_predict+stab_within_cond+response_timing_first+response_timing_second+ plot_layout(guides = "collect", ncol=2)+
  plot_annotation(tag_levels = 'A')
sup_plots

ggsave(file.path(out_folder,'sfig3_individuality.png'),sup_plots, width = 12, height =12)
ggsave(file.path(out_folder,'sfig3_individuality.pdf'),sup_plots,  width = 12, height =12)