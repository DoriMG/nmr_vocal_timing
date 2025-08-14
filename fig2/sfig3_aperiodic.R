library(ggplot2)
library(patchwork)

data_folder = "data"
out_folder = "figs"


# Figure S3A-C

data_file = file.path(data_folder, "results_AP_predictable.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

df_temp = df[df$noise  == 300,]
ms_300 = ggplot(df_temp, aes(x=time, y=value)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#B7B597') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#B7B597')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,0.6))+ggtitle('300 ms')
ms_300

df_temp = df[df$noise  == 600,]
ms_600 = ggplot(df_temp, aes(x=time, y=value)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#6B8A7A') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#6B8A7A')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.2))+ggtitle('600 ms')
ms_600

df_temp = df[df$noise  == 900,]
ms_900 = ggplot(df_temp, aes(x=time, y=value)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#254336') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#254336')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('900 ms')
ms_900


# Figure S3D
data_file = file.path(folder, "hist_periodic_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

call_delay_periodic = ggplot(df, aes(y=data, x=condition, fill=condition))+  
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Mean call delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.8))
call_delay_periodic

lmm = lmer(data  ~condition  +(1|animal), data =df)
anova(lmm)


# Figure S3E
data_file = file.path(data_folder, "AP_peak_delay.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$noise = factor(df$noise)

call_delay_peak = ggplot(df, aes(y=data_sec, x=noise , fill=noise ))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Peak delay (s)',x= 'Condition')+ theme_classic()
call_delay_peak

lmm = lmer(data_sec  ~noise  +(1|dataset), data =df)
anova(lmm)

# Figure S3F
data_file = file.path(folder, "callnum_AP_predictable.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data*100 # convert to percentage

callnum_per_epoch = ggplot(df, aes(y=data, x=condition, fill=condition))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='% epochs with call',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,50))
callnum_per_epoch

lmm = lmer(data  ~condition  +(1|dataset), data =df)
anova(lmm)


## Unpredictable periodic ##
# Figure S3G-I

data_file = file.path(data_folder, "results_AUP.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

df_temp = df[df$noise  == 300,]
ms_300_AUP = ggplot(df_temp, aes(x=time, y=value)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#B7B597') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#B7B597')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.8))+ggtitle('300 ms')
ms_300_AUP

df_temp = df[df$noise  == 600,]
ms_600_AUP = ggplot(df_temp, aes(x=time, y=value)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#6B8A7A') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#6B8A7A')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.8))+ggtitle('600 ms')
ms_600_AUP

df_temp = df[df$noise  == 900,]
ms_900_AUP = ggplot(df_temp, aes(x=time, y=value)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#254336') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#254336')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('900 ms')
ms_900_AUP


# Figure S3J

data_file = file.path(folder, "hist_AUP.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$noise = factor(df$noise)

call_delay_periodic_AUP = ggplot(df, aes(y=data, x=noise , fill=noise ))+  
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Mean call delay (s)',x= 'Condition')+ theme_classic()+coord_cartesian(ylim=c(0,0.8))
call_delay_periodic_AUP

lmm = lmer(data  ~noise  +(1|animal), data =df)
anova(lmm)


# Figure S3K
data_file = file.path(folder, "AUP_peak_delay.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$noise = factor(df$noise)

call_delay_peak_AUP = ggplot(df, aes(y=data_sec, x=noise, fill=noise))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Peak delay (s)',x= 'Condition')+ theme_classic()
call_delay_peak_AUP

lmm = lmer(data_sec  ~noise  +(1|dataset  ), data =df)
anova(lmm)

# Figure S3L
data_file = file.path(folder, "callnum_AUP.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data * 100 # convert to percentage
df$noise = factor(df$noise)

callnum_periodic_AUP = ggplot(df, aes(y=data, x=noise, fill=noise))+ 
  stat_summary(fun=mean, geom='bar', alpha=1,  fill=c('#B7B597', '#6B8A7A','#254336'))+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='% stimuli with response',x= 'Condition')+ theme_classic()
callnum_periodic_AUP

lmm = lmer(data  ~noise  +(1|animal), data =df)
anova(lmm)




