library(ggplot2)
library("dplyr")
library(smplot2)
library(rstatix)
library(emmeans)

data_folder = "data"
out_folder = "figs"

# Figure 3A
data_file = file.path(data_folder, "strategy_per_animal.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

df$percentage  = df$percentage *100
df$percentage_all  = df$percentage_all *100
df$condition = factor(df$condition)
df$epoch_factor = factor(df$epoch_type)

perc_onset_v_offset = ggplot(data=df, aes(x=epoch_type   , y=percentage, fill=condition)) +
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position = position_dodge(width=0.8))+
  scale_x_continuous(breaks=c(1,2,3), labels=c("Onset","Offset", "Both")) + 
  scale_fill_manual(values=c('#6B8A7A', '#254336'))+
  labs(y ='Epochs (%)', x=NULL)+theme_classic()+  facet_wrap(~experiment)
perc_onset_v_offset


# Stats noise experiment
lmm = lmer(percentage  ~epoch_factor*condition +(1|animal ), data =df[df$experiment=='noise',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")

# Stats sc experiment
lmm = lmer(percentage  ~epoch_factor*condition +(1|animal ), data =df[df$experiment=='sc',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")



## Fig 3B -Stability across conditions
data_file = file.path(data_folder, "stab_across_conditions.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

stab_across_cond = ggplot(data=df, aes(x=noise_perc_600 , y=noise_perc_900 )) +
  geom_point()+
  labs(x ='Calls during noise (600ms)', y='Calls during noise (900ms)')+
  sm_statCorr(color='black', corr_method='pearson')+
  theme_classic()+facet_wrap(~experiment)
stab_across_cond


## Fig3C -  Timing in session
data_file = file.path(data_folder, "timing_in_session.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$condition = factor(df$condition)
df$time_perc = df$time*10+5
df$data = df$data*100

time_in_session = ggplot(df, aes(x=time_perc, y=data, col=condition, fill=condition)) +  
  stat_summary(fun=mean, geom='line', alpha=1) +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, colour=NA)+
  labs(y='Calls (%)',x= 'Time (% of session)', fill=NULL, col=NULL)+ theme_classic()+
  scale_color_manual(values=c('#6B8A7A', '#254336'), labels=c('600 ms', '900 ms'))+
  scale_fill_manual(values=c('#6B8A7A', '#254336'), labels=c('600 ms', '900 ms'))+
  facet_wrap(~experiment)
time_in_session

# STats
lmm = lmer(data  ~time_perc*condition +(1|session ), data =df[df$experiment=='noise',])
anova(lmm)
EMM <- emmeans(lmm, ~ time_perc*condition)
test(pairs(EMM,  by = "condition"), by = NULL, adjust = "bh")

lmm = lmer(data  ~time_perc*condition +(1|session ), data =df[df$experiment=='sc',])
anova(lmm)
EMM <- emmeans(lmm, ~ time_fac*condition)
test(pairs(EMM,  by = "condition"), by = NULL, adjust = "bh")


## Fig 3D - Call response timing

data_file = file.path(data_folder, "delay_stats.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$condition = factor(df$condition)

df_temp = df[df$epoch_type<3,]
df_temp$epoch_type = factor(df_temp$epoch_type)

response_timing = ggplot(data=df_temp, aes(x=epoch_type  , y=mean_response_base , fill=condition)) +
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position = position_dodge(width=0.8))+
  scale_x_discrete(breaks=c(1,2), labels=c("Onset","Offset")) + 
  scale_fill_manual(values=c('#6B8A7A', '#254336'))+
  labs(y ='Response time (s)', x=NULL)+theme_classic()+  facet_wrap(~experiment)
response_timing


## Stats
df$epoch_factor = factor(df$epoch_type)
lmm = lmer(mean_response_base  ~epoch_factor*condition +(1|animal )+ (1|colony), data =df[df$experiment=='noise',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")

lmm = lmer(mean_response_base  ~epoch_factor*condition +(1|animal ), data =df[df$experiment=='sc',])
anova(lmm)
EMM <- emmeans(lmm, ~ epoch_factor*condition)
test(pairs(EMM, by = "epoch_factor"), by = NULL, adjust = "bh")


## Final plots
all_plots = perc_onset_v_offset+stab_across_cond+time_in_session+response_timing+ plot_layout(guides = "collect")+
  plot_annotation(tag_levels = 'A')
all_plots
ggsave(file.path(out_folder,'fig3_individuality.png'),all_plots, width = 12, height =8)
ggsave(file.path(out_folder,'fig3_individuality.pdf'),all_plots,  width = 12, height =8)


