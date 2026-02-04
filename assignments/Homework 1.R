library(reader)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)

#load texts
getwd()
setwd("/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/texts")
list.files(recursive = TRUE)
file <-"/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/texts"
file.exists(file)
file_a <- "A07594__Circle_of_Commerce.txt"
file_b <- "B14801__Free_Trade.txt"
list.files()
text_a <- readLines(file_a)
text_b <- readLines(file_b)
texts <- tibble(
  doc_title = c("text a", "text b"),
  text = c(text_a, text_b))
texts

#Create a tibble called corpus_diagnostics
corpus_diagnostics <- texts %>% 
  mutate(n_chars = str_length(text)) %>% 
  unnest_tokens(word, text) %>%
  mutate(word = str_to_lower(word)) %>%
  group_by(doc_title) %>%
  summarize(
    n_chars = first(n_chars),
    n_word_tokens = n(),
    n_word_types =n_distinct(word)
  )
print(corpus_diagnostics)

#Raw word counts and frequency before stopwords removal
raw_freq <- texts %>% 
  mutate(n_chars = str_length(text)) %>% 
  unnest_tokens(word, text) %>%
  count(doc_title,word,sort=TRUE) %>% 
  mutate(raw_freq=n/n_distinct(word))
print(raw_freq)

#Define stopwords
data("stop_words")
custom_stopwords <- tibble(word = c(  "vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
                                      "beene","thereof","thus"))
all_stopwords <- bind_rows(stop_words, custom_stopwords)%>%
  distinct(word)
all_stopwords %>% slice(1:10)

#Remove stopwords
word_counts <- texts %>%
  unnest_tokens(word, text) %>% 
  mutate(word=str_to_lower(word)) %>% 
  anti_join(all_stopwords, by="word") %>% 
  count(doc_title,word, sort=TRUE)
word_counts

# Total number of words(after stopwords removal)
doc_lengths <- word_counts %>%
  group_by(doc_title) %>% 
  summarize(word_n=sum(n))
print(doc_lengths)

#Normalization: Add a column to show relative frequency
word_counts_normalized <- doc_lengths %>% 
  left_join(word_counts,by="doc_title") %>% 
  mutate(
    n= as.numeric(n),
    word_n= as.numeric(word_n),
    relative_freq=n/word_n)
print(word_counts_normalized)

#Filter "trade"
trade_compare <- word_counts_normalized %>% 
  filter(word == "trade") %>%
  left_join(raw_freq)
print(trade_compare)

#Visualization
plot_n_words <- 20
word_comparison_tbl <- word_counts %>%
  pivot_wider(
    names_from = doc_title,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(max_n = pmax(`text a`, `text b`)) %>%
  arrange(desc(max_n)) %>% 
  slice_head(n = plot_n_words)
word_comparison_tbl

word_plot_data <- word_counts_normalized %>%
  filter(word %in% word_comparison_tbl$word) %>%
  select(word, doc_title, relative_freq) %>%
  mutate(word = fct_reorder(word, relative_freq, .fun = max))
word_plot_data

ggplot(word_plot_data, aes(x = relative_freq, y = word)) +
  geom_col() +
  facet_wrap(~ doc_title, scales = "free_x") +
  labs(
    title = "Words Relative Frequencies (stopwords removed)",
    subtitle = paste0(
      "Top ", plot_n_words,
      " words by maximum relative frequency across both texts"
    ),
    x = "Relative frequency",
    y = NULL
  ) +
  theme_minimal()

nrow(word_comparison_tbl)
dplyr::n_distinct(word_comparison_tbl$word)

word_plot_data %>%
  dplyr::distinct(word) %>%
  nrow()