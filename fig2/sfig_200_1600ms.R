library(ggplot2)
library(patchwork)
library("dplyr")
library(lme4)
library(lmerTest)

data_folder = "data"
out_folder = "figs"


data_file = file.path(data_folder, "data_long_short_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure 2A - 1600ms noise response
df_temp = df[df$condition == '1600noise',]
ms_1600 = ggplot(df_temp, aes(x=time_sec, y=data)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#EA2264') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#EA2264')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.8))+ggtitle('1600 ms')
ms_1600

# Figure 2B - 200ms noise response
df_temp = df[df$condition == '200noise',]
ms_200 = ggplot(df_temp, aes(x=time_sec, y=data)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#640D5F') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#640D5F')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.8))+ggtitle('200 ms')
ms_200


## Properties

data_file = file.path(data_folder, "callnum_p_epoch_long_short_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data*100 # convert to percentage

# Figure S2A
callnum_per_epoch = ggplot(df, aes(y=data, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#EA2264', '#640D5F'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='% epochs with call',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,50))
callnum_per_epoch

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)

data_file = file.path(data_folder, "hist_long_short_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2B
call_delay_periodic = ggplot(df, aes(y=data, x=condition, fill=condition))+  
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#EA2264', '#640D5F'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Mean call delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.8))
call_delay_periodic

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)

df %>%
  group_by(condition) %>%
  summarise(disp = mean(data))

data_file = file.path(data_folder, "long_short_noise_peak_delay.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2C
call_delay_peak = ggplot(df, aes(y=data_sec, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#EA2264', '#640D5F'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Peak delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.6))
call_delay_peak

df %>%
  group_by(condition) %>%
  summarise(disp = mean(data_sec))

lmm = lmer(data  ~condition  +(1|session ), data =df)
anova(lmm)

all_plots = (ms_1600|ms_200)/(callnum_per_epoch|call_delay_periodic|call_delay_peak)


ggsave(file.path(out_folder,"sfig4_long_short.png"),all_plots, width = 12, height = 8)
ggsave(file.path(out_folder,"sfig4_long_short.pdf"),all_plots, width = 12, height = 8)
