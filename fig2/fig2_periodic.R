library(ggplot2)
library(patchwork)
library("dplyr")

data_folder = "data"
out_folder = "figs"


data_file = file.path(data_folder, "data_periodic_noise.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure 2A - 300ms noise response
df_temp = df[df$condition == '300ms',]
ms_300 = ggplot(df_temp, aes(x=time_sec, y=data)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#B7B597') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#B7B597')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,0.6))+ggtitle('300 ms')
ms_300

# Figure 2B - 600ms noise response
df_temp = df[df$condition == '600ms',]
ms_600 = ggplot(df_temp, aes(x=time_sec, y=data)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#6B8A7A') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#6B8A7A')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  xlim(c(0,1.2))+ggtitle('600 ms')
ms_600

# Figure 2C - 900ms noise response
df_temp = df[df$condition == '900ms',]
ms_900 = ggplot(df_temp, aes(x=time_sec, y=data)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#254336') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#254336')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('900 ms')
ms_900



### SC data ####

data_file = file.path(data_folder, "data_900_sc.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Figure 2G - 300ms SC response
df_temp = df[df$condition == '300sc',]
sc_300 = ggplot(df_temp, aes(x=time_sec, y=data)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#B7B597') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#B7B597')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('300 ms')+
  xlim(c(0,0.6))
sc_300

# Figure 2H - 600ms SC response
df_temp = df[df$condition == '600sc',]
sc_600 = ggplot(df_temp, aes(x=time_sec, y=data)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#6B8A7A') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#6B8A7A')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('600 ms')+
  xlim(c(0,1.2))
sc_600

# Figure 2I - 900ms SC response
df_temp = df[df$condition == '900sc',]
sc_900 = ggplot(df_temp, aes(x=time_sec, y=data)) +  
  stat_summary(fun=mean, geom='line', alpha=1, color='#254336') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#254336')+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+ggtitle('900 ms')
sc_900

all_plots = ms_300 +ms_600+ ms_900 +
               sc_300 + sc_600 + sc_900 + plot_layout(ncol=3, widths = c(3,3,3))+ plot_annotation(tag_levels = 'A')
all_plots


ggsave(file.path(out_folder,"fig2_periodic.png"),all_plots, width = 12, height = 8)
ggsave(file.path(out_folder,"fig2_periodic.pdf"),all_plots, width = 12, height = 8)
