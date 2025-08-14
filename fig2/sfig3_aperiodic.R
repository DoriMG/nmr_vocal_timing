library(ggplot2)
library(patchwork)

data_folder = "data"
out_folder = "figs"


# Figure S3A-C


data_file = file.path(folder, "callnum_p_epoch_periodic_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data*100 # convert to percentage

# Figure S2A
callnum_per_epoch = ggplot(df, aes(y=data, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='% epochs with call',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,50))
callnum_per_epoch

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)

data_file = file.path(folder, "hist_periodic_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2B
call_delay_periodic = ggplot(df, aes(y=data, x=condition, fill=condition))+  
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Mean call delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.8))
call_delay_periodic

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)


data_file = file.path(data_folder, "periodic_noise_peak_delay.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2C
call_delay_peak = ggplot(df, aes(y=data_sec, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Peak delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.6))
call_delay_peak

lmm = lmer(data  ~condition  +(1|session ), data =df)
anova(lmm)


## SC ##

data_file = file.path(folder, "callnum_p_epoch_900_sc.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data * 100 # convert to percentage

# Figure S2D
callnum_periodic_sc = ggplot(df, aes(y=data, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='% stimuli with response',x= 'Condition')+ theme_classic()
callnum_periodic_sc

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)


data_file = file.path(folder, "hist_data_900_sc.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2E
call_delay_periodic_sc = ggplot(df, aes(y=data, x=condition, fill=condition))+  
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Mean call delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.8))
call_delay_periodic_sc

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)


data_file = file.path(folder, "data_sc_peak_delay.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure S2F
call_delay_peak_sc = ggplot(df, aes(y=data_sec, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Peak delay (s)',x= 'Condition')+ theme_classic()
call_delay_peak_sc

lmm = lmer(data  ~condition  +(1|session ), data =df)
anova(lmm)

all_plots = callnum_per_epoch+call_delay_periodic+call_delay_peak+
  callnum_periodic_sc+call_delay_periodic_sc+call_delay_peak_sc+ 
  plot_layout(ncol=3, widths = c(3,3,3))+ plot_annotation(tag_levels = 'A')


ggsave(file.path(out_folder,"sfig2_callnums.png"),all_plots, width = 8, height = 6)
ggsave(file.path(out_folder,"sfig2_callnums.pdf"),all_plots, width = 8, height = 6)
