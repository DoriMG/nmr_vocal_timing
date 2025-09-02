library(ggplot2)
library("data.table") 
library(patchwork)

folder = "data"
out_folder = "figs"


data_file = file.path(folder, "all_ici.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df= df_playback
df$playback = 1

data_file = file.path(folder, "response_time_by_bout_hierarchy.csv")
df_hier <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df_hier$playback = 0

df = bind_rows(df, df_hier)



response_time = ggplot(df, aes(x=ici, fill=factor(playback))) + 
  geom_histogram(aes(y = after_stat(count / sum(count))),binwidth=0.1) +
  scale_y_continuous(labels = scales::percent)+
  scale_fill_manual(values=c('#E0B0D5', '#68B0AB'), labels = c("Hierarchy", "Playback"))+
  facet_wrap(~playback,  ncol=1,)+
  labs(x='Response time (s)', y='Number of calls')+ theme_classic()+
  xlim(-0.3,3)
response_time

df_temp = df[df$ici<3,]
response_time_comp  = ggplot(df_temp, aes(x=factor(playback), y=ici, fill = factor(playback))) + 
  geom_boxplot()+
  scale_fill_manual(values=c('#E0B0D5','#68B0AB'), labels = c("Playback", "Hierarchy"))+
  scale_x_discrete(labels=c( 'Hierarchy', 'Playback'))+
  labs(y='Response time (s)', x=NULL, fill='Experiment')+ theme_classic()

response_time_comp

df_temp %>%
  group_by(factor(playback)) %>%
  summarise(mean = mean(ici), sd = sd(ici))

t.test(df_temp$ici[df_temp$playback==0], df_temp$ici[df_temp$playback==1])


# Fig 1C&F Interruptions
data_file = file.path(folder, "interruptions.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

# Fig 1C Interruptions 2 animal
df_hier = df[df$playback==0,]
interrupt_hier = ggplot(df_hier, aes(x=factor(shuff), y=data)) + 
  stat_summary(fun=mean, geom='bar', alpha=1, fill=c('#E0B0D5','#808080')) +
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  scale_x_discrete(breaks=c(0,1),
                   labels=c('Two-animal', 'Shuffe')) + 
  labs(y='Calls interrupted (%)',x=NULL)+ theme_classic()+ coord_cartesian(ylim = c(0, 0.25))
interrupt_hier

t.test(df_hier$data[df_hier$shuff==0], df_hier$data[df_hier$shuff==1])

# Fig 1C Interruptions 1 animal
df_pb = df[df$playback==1,]
interrupt_pb = ggplot(df_pb, aes(x=factor(shuff), y=data)) + 
  stat_summary(fun=mean, geom='bar', alpha=1, fill=c('#68B0AB','#808080')) +
  scale_x_discrete(breaks=c(0,1),
                   labels=c('Playback', 'Shuffe')) + 
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  labs(y='Calls interrupted (%)',x=NULL)+ theme_classic()+ coord_cartesian(ylim = c(0, 0.25))
interrupt_pb
t.test(df_pb$data[df_pb$shuff==0], df_pb$data[df_pb$shuff==1])

#Fig 1G Calls during behavior vs random
data_file = file.path(folder, "perc_calls_during_behavior.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$perc = df$perc*100

perc_calls_during_behavior  = ggplot(df, aes(x=shuffle, y=perc, fill = factor(shuffle))) + 
  stat_summary(fun=mean, geom='bar', alpha=1)+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  scale_fill_manual(values=c('#E0B0D5','grey'), labels = c("Data", "Shuffled"))+
  scale_x_discrete(labels=c("Data", "Shuffled"))+
  labs(y='Calls during touch behavior (%)', x=NULL, fill=NULL)+ theme_classic()+ theme(legend.position="none")
perc_calls_during_behavior

t.test(df$perc[df$shuff=='Data'], df$perc [df$shuff=='Shuffle'])

#Fig 1H Calls during 6 different behaviors
data_file = file.path(folder, "perc_calls_by_behavior.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)
df$data = df$data*100

perc_calls_by_behavior  = ggplot(df, aes(x=reorder(touch_type,-data), y=data, fill = factor(touch_type))) + 
  stat_summary(fun=mean, geom='bar', alpha=1)+
  stat_summary(fun.data = mean_cl_normal, geom="errorbar", width=0.3)+
  scale_x_discrete(labels=c("Snout-to-snout", "No touch", "Body-to-body", "Snout-to-body", "Anogenital", "Passing", "Other"))+
  labs(y='Calls during touch behavior (%)', x='Type of interaction', fill=NULL)+ theme_classic()+ theme(legend.position="none")
perc_calls_by_behavior


# Save out all
ggsave(file.path(save_folder,"fig1_calls_during_by_behavior.pdf"),all_plots, width = 8, height =4)
ggsave(file.path(save_folder,"fig1_calls_during_by_behavior.png"),all_plots, width = 8, height = 4)

