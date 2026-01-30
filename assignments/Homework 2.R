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


#1. Raw Word Counts
#Tokenize and clean the text
data("stop_words")
custom_stopwords <- tibble(word = c("vnto", "haue", "doo", "hath", "bee", "ye", "thee"))
all_stopwords <- bind_rows(stop_words, custom_stopwords)%>%
  distinct(word)
all_stopwords %>% slice(1:10)

word_normalized <- texts %>%
  unnest_tokens(word, text) %>% 
  mutate(word=str_to_lower(word)) %>% 
  anti_join(all_stopwords, by="word") %>% 
  count(doc_title,word, sort=TRUE)
print (word_normalized)

# Introduce Bing sentiment dictionary
bing <- get_sentiments("bing")

# Compute raw sentiment totals
raw_sentiment <- word_normalized %>%
  inner_join(bing, by = "word") %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  group_by(doc_title) %>% 
  summarise(
  n_positive = sum(positive),
  n_negative = sum(negative),
  net_sentiment = n_positive - n_negative
)
print (raw_sentiment)


#II. TF-IDF–Weighted Sentiment Analysis
#Compute TF-IDF for words in each document
texts <- c(
  "Circle_of_Commerce" = text_a,
  "Free_Trade"= text_b)
corp <- corpus(texts)




