library(ggplot2)
library(patchwork)
library("dplyr")
library(viridis)

data_folder = "data"
out_folder = "figs"

# Load data from fig 2
data_file = file.path(data_folder, "data_long_short_noise_perc.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Load model responses
data_file = file.path(data_folder, "model_responses_long_short.csv")
df_model <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df_model$model <- factor(df_model$model, levels = c("ramp", "noise", "feedback"))

# Fig 4B, F, J
df_temp = df[df$condition == '200noise',]
df_model_temp = df_model[df_model$stim_len == 200,]
model_200ms = ggplot(df_model_temp, aes(x=time, y=data)) +
  stat_summary(data=df_temp, aes(time_sec, y=data), fun=mean, geom='line', alpha=1, color='grey') +
  stat_summary(data=df_temp, aes(time_sec, y=data),fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='grey')+
  stat_summary(fun=mean, geom='line', alpha=1, color='#640D5F') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#640D5F')+
  labs(y='Proportion of calls',x= 'Time (s)')+ theme_classic()+facet_wrap(~model, ncol=1)+
  xlim(c(0,1.8))+ggtitle('200 ms')
model_200ms



# Fig 4C, G, K
df_temp = df[df$condition == '1600noise',]
df_model_temp = df_model[df_model$stim_len == 1600,]
model_1600ms = ggplot(df_model_temp, aes(x=time, y=data)) +
  stat_summary(data=df_temp, aes(time_sec, y=data), fun=mean, geom='line', alpha=1, color='grey') +
  stat_summary(data=df_temp, aes(time_sec, y=data),fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='grey')+
  stat_summary(fun=mean, geom='line', alpha=1, color='#EA2264') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#EA2264')+
  labs(y='Proportion of calls',x= 'Time (s)')+ theme_classic()+facet_wrap(~model, ncol=1)+
  xlim(c(0,1.8))+ggtitle('1600 ms')
model_1600ms

all_fitted = model_1600ms|model_200ms
all_fitted 


## Fig 4M - model error
data_file = file.path(data_folder, "model_errors_long_short.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$stim_len = factor(df$stim_len)
df$model <- factor(df$model, levels = c("ramp", "noise", "feedback"))
mean_errors = ggplot(df, aes(y=error_p_ani, x=stim_len, fill=model ))+ 
  stat_summary(fun=mean, geom='bar', alpha=1, position = position_dodge(width=0.8), width=0.8) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3, position = position_dodge(width=0.8)) +
  geom_point(position=position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2), alpha=0.5, stroke = 0,shape=16,size=1) + 
  scale_fill_manual(values=c('#26547c', '#ef476f', '#ffd166'))+
  labs(y='Error (sse)',x= NULL, fill=NULL)+ theme_classic()
mean_errors

library(rcompanion)

scheirerRayHare(error_p_ani ~ model*stim_len,
                data = df)




all_plots = (all_fitted/(mean_errors+plot_spacer()))+ plot_layout( heights = c(3,1))
all_plots
ggsave(file.path(out_folder,'fig4_model_long_short.png'),all_plots, width = 8, height =8)
ggsave(file.path(out_folder,'fig4_model_long_short.pdf'),all_plots, width = 8, height =8)
