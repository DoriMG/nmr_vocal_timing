
library(ggplot2)
library("data.table") 
library(patchwork)

folder = "data"
out_folder = "figs"


## Fig S1 Calls after touch
data_file = file.path(folder, "calls_after_behavior.csv")
df <- read.csv(data_file, header=TRUE, stringsAsFactors=TRUE)

touch_type = ggplot(df, aes(x=time_sec, y=data)) + 
  stat_summary(fun=mean, geom='line', alpha=1, color='#6A0066') +
  stat_summary(fun.data = mean_cl_normal, geom="ribbon", alpha=0.5, fill='#6A0066')+
  geom_vline(xintercept = 0, linetype="dashed", size=1.2)+
  labs(y='Number of calls',x= 'Time (s)')+ theme_classic()+
  facet_wrap(~touch_type, ncol=3)+
  xlim(c(-3,3))
touch_type


# Save out all
ggsave(file.path(out_folder,"touch_type.pdf"),touch_type, width = 6, height =4)
ggsave(file.path(out_folder,"touch_type.png"),touch_type, width = 6, height = 4)
