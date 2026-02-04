library(reader)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)
library(quanteda)

#load texts
getwd()
setwd("/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/texts")
file <-"/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/texts"
file.exists(file)
file_a <- "A07594__Circle_of_Commerce.txt"
file_b <- "B14801__Free_Trade.txt"
text_a <- readLines(file_a)
text_b <- readLines(file_b)
texts <- tibble(
  doc_title = c("text a", "text b"),
  text = c(text_a, text_b))


#Section I. Raw Word Counts
#Tokenize and clean the text
data("stop_words")
custom_stopwords <- tibble(word = c("vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
                                    "beene","thereof","thus"))
all_stopwords <- bind_rows(stop_words, custom_stopwords)%>%
  distinct(word)

word_normalized <- texts %>%
  unnest_tokens(word, text) %>% 
  mutate(word=str_to_lower(word)) %>% 
  anti_join(all_stopwords, by="word") %>% 
  count(doc_title,word, sort=TRUE)

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


#Section II. TF-IDF–Weighted Sentiment Analysis
#4. Compute TF-IDF for words in each document
word_tf_idf <- word_normalized %>%
  bind_tf_idf(word, doc_title, n)
print(word_tf_idf)

#5 Keep only sentiment-bearing words
bing_tf_idf <- word_tf_idf %>%
  inner_join(bing, by = "word")
print(bing_tf_idf)

#6 Compute TF-IDF–weighted sentiment totals
sentiment_tfidf_summary <- bing_tf_idf %>%
  group_by(doc_title) %>%
  summarise(
    tfidf_positive = sum(tf_idf[sentiment == "positive"], na.rm = TRUE),
    tfidf_negative = sum(tf_idf[sentiment == "negative"], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    tfidf_positive = tidyr::replace_na(tfidf_positive, 0),
    tfidf_negative = tidyr::replace_na(tfidf_negative, 0),
    net_sentiment_tfidf = tfidf_positive - tfidf_negative
  )

#7: Compare Raw and TF-IDF Sentiment
final_sentiment_comparison <- raw_sentiment %>%
  left_join(sentiment_tfidf_summary)

write.csv(
  final_sentiment_comparison,
  "compare.csv")

read.csv("compare.csv")