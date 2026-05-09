library(tidyverse)
library(scales)
library(ggrepel)

theme_set(theme_minimal())

df_trajectory <- read_csv("data/processed/trajectory_s50.csv")

max_ep <- max(df_trajectory$episode, na.rm = TRUE)

labels_df <- df_trajectory %>%
  filter(episode == max_ep) %>%
  arrange(desc(win_probability)) %>%
  slice_head(n = 5) %>%
  mutate(label = paste0(castaway, " [", percent(win_probability), "]"))

plot <- df_trajectory %>%
  ggplot(aes(x = episode, y = win_probability, color = castaway, group = castaway)) +
  geom_line() +
  ylim(0, 1) +
  scale_x_continuous(breaks = seq(0, max_ep)) +
  geom_point(data = labels_df, size = 1.7, show.legend = FALSE) +
  ggrepel::geom_text_repel(
    data = labels_df,
    aes(x = episode, y = win_probability, label = label, color = castaway),
    nudge_x = 0.5,
    hjust = -1,
    direction = "y",
    segment.size = 0.25,
    show.legend = FALSE
  ) +
  coord_cartesian(xlim = c(0, max_ep + 2))

ggsave("reports/trajectory_s50_r.png", width = 10, height = 6)
