library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)
library(quanteda)
library(quanteda.textstats)
library(purrr)
library(readr)




#Load texts
setwd("/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/Data exploration_Text Files")
files <- list.files("/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/Data exploration_Text Files",
                    pattern = "\\.txt$",
                    full.names = TRUE)
texts <- character(length(files))
for (i in 1:length(files)) {
  texts[i] <- paste(readLines(files[i], warn = FALSE),
                    collapse = " ")
}
texts_df <- data.frame(
  doc_title = basename(files),
  text = texts,
  stringsAsFactors = FALSE
)
doc_map <- data.frame(
  doc_id = 1:length(files),
  filename = basename(files),
  stringsAsFactors = FALSE
)
print(doc_map)




#Step 0: Tidy the text
#1. Remove the Long ſ and s
text_tbl <- texts_df %>%
  mutate(text_clean = str_replace_all(text,c(
      "ſ"="s",
      "2dly" = "secondly",
      "vpon" = "upon"))
      )
#2. Basic Cleaning:remove stop words and symbols
corp <- corpus(text_tbl,text_field = "text_clean")
toks <- tokens(
  corp,
  remove_punct   = TRUE,
  remove_numbers = TRUE,
  remove_symbols = TRUE
)
toks <- tokens_tolower(toks)
custom_stop <- c(
  "vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
  "beene","thereof","thus","ll","ss"
)
toks <- tokens_remove(
  toks,
  pattern = c(stopwords("en"), custom_stop)
  )
toks <- tokens_select(
  toks,
  pattern = "^.$",
  selection = "remove",
  valuetype = "regex"
)




#Approach 1: TF-IDF Lexical Distinctiveness
#1. Document-feature matrix (DFM)
dfm_mat <- dfm(toks)
#2.Compute TF–IDF weight
dfm_tfidf <- dfm_tfidf(dfm_mat)
#3.Top 15 TF-IDF Terms
top15_df <- dfm_tfidf %>%
  tidy() %>%
  group_by(document) %>%
  arrange(desc(count),.by_group = TRUE) %>%
  slice_head(n = 15)
print(top15_df)





#Approach 2:Pearson correlation: similarity and distance between texts
#1.Trimming very rare words from the DFM
dfm_mat <- dfm_trim(dfm_mat, min_termfreq = 5)
#2.Compute pairwise Pearson correlations
sim_r <- textstat_simil(
  dfm_mat,
  method = "correlation",
  margin = "documents")
sim_r
#3. Convert matrix to be a long format data frame
r_mat <- as.matrix(sim_r)
heat_df <- as.data.frame(r_mat) %>%
  rownames_to_column("doc_i") %>%
  pivot_longer(-doc_i, names_to = "doc_j", values_to = "r")
#4 Finding the Most similar and least similar pairs
MostS_Pair <- heat_df %>%
  filter(doc_i != doc_j) %>%
  arrange(desc(r)) %>%
  head(16)
print(MostS_Pair)
LeastS_Pair <- heat_df %>%
  filter(doc_i != doc_j) %>%
  arrange(r) %>%
  head(16)
print(LeastS_Pair)
#5. Visualization: Heat map
ggplot(heat_df, aes(x = doc_j, y = doc_i, fill = r)) +
  geom_tile() +
  coord_fixed() +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0
  ) +
  labs( title = "Pearson Correlation Between Documents",
        x = NULL,
        y = NULL,
        fill = "Correlation"
  ) + theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )





#Approach 3:Syntactic complexity profile
#0-1. Choose two Text and Load them
setwd("/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/Data exploration_Text Files")
file <-"/Users/jifei/Desktop/Duke/26 Spring/IDS 570 Text as Data/texts"
text17 <- read_file("A32839.txt")
text20 <- read_file("A69858.txt")
texts_df <- tibble(
  document = c("text17", "text20"),
  text = c(text17, text20))
#0-2.Clean Data (Because I have tokenized them before, so I redo my data clean here)
data("stop_words")
custom_stopwords <- tibble(
  word = c("vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
                                      "beene","thereof","thus","ll","ss"))
all_stopwords <- bind_rows(stop_words, custom_stopwords) %>%
  distinct(word)
#1.#Load an English UD model(I have downloaded it once)
library(udpipe)
model_info <- udpipe_download_model(language = "english-ewt")
ud_model <- udpipe_load_model(model_info$file_model)
#2. Processing the Text:annotate both texts using UDPipe
anno_df <- texts_df %>%
  mutate(anno = map2(text, document, ~ udpipe_annotate(ud_model, x = .x, doc_id = .y) %>%
                  as.data.frame())
  ) %>%
  select(anno) %>%
  unnest(anno) %>%
  rename(document = doc_id) %>%
  select(
    document,
    paragraph_id,
    sentence_id,
    token_id,
    token,
    lemma,
    upos,                 # part of speech
    feats,                # grammatical features (e.g., verb form)
    head_token_id,        # head of dependency relation
    dep_rel               # dependency relation type
  )
anno_df %>% glimpse()
#3. Create an example parsed sentence
example_sentence <- tibble(
  token = c("The", "big", "dog", "barks"),
  token_id = c(1, 2, 3, 4),
  head_token_id = c(3, 3, 4, 0),
  Relationship = c(
    '"The" depends on word #3 (dog)',
    '"big" depends on word #3 (dog)',
    '"dog" depends on word #4 (barks)',
    '"barks" is the ROOT (doesn\'t depend on anything)'
  )
)
example_sentence %>%
  knitr::kable(
    caption = 'Example: Dependency structure of "The big dog barks"',
    align = c("l", "c", "c", "l")
  )
#4.Processing the grammar
syntax_df <- anno_df %>%
  mutate(
    #is it a word (and not punctuation?)
    is_word = upos != "PUNCT",
    # Is this an independent clause? finite verbs are proxy for independent clauses
    is_clause = (upos %in% c("VERB", "AUX")) & str_detect(coalesce(feats, ""), "VerbForm=Fin"),
    # Dependent clause? 
    is_dep_clause = dep_rel %in% c(
      "advcl",       #adverbial clause 
      "ccomp",       #clausal complement
      "xcomp",       #open clausal complement
      "acl",         #adnomial clause
      "acl:relcl"    #relative clause
    ),
    # Is this coordination?
    is_coord = dep_rel %in% c("conj", "cc"),
    # Nominal complexity: complex noun phrases
    is_complex_nominal = dep_rel %in% c(
      "amod",        # adjective modifier
      "nmod",        #nominal modifier
      "compound",    #compound
      "appos"        #apposition
    )
    )
#5. Measuring the Sentences
sentence_df <- syntax_df %>%
  filter(is_word) %>%                   #count words (not punctuation)
  group_by(document, sentence_id) %>%   #group by document and sentence
  summarise(
    words= n(),                         #number of words per sentence
    clauses = sum(is_clause),           #number of clauses per sentence
    dep_clauses = sum(is_dep_clause),   #number of dependent clauses per sentence
    .groups = "drop"
  )
print(sentence_df)
#6-1.Calculate Mean Length of Sentence (MLS)
mls_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    MLS = mean(words),                  # Average words per sentence
    .groups = "drop"
  )
print(mls_df)
#6-2. Calculate clauses per sentence (C/S)
clausal_density_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    sentences = n(),
    clauses   = sum(clauses),
    C_per_S   = clauses / sentences,
    .groups = "drop"
  )
print(clausal_density_df)
#6-3. Dependent Clauses per Clause and/or Sentence (DC/C and DC/S)
subordination_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    clauses = sum(clauses),
    dep_clauses = sum(dep_clauses),
    sentences = n(),
    DC_per_C = dep_clauses / pmax(clauses, 1),
    DC_per_S = dep_clauses / sentences,
    .groups = "drop"
  )
print(subordination_df)
#6-4.Coordination per Clause and/or Sentence
coordination_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    coord_relations = sum(is_coord),
    clauses         = sum(is_clause),
    sentences       = n_distinct(sentence_id),
    Coord_per_C     = coord_relations / pmax(clauses, 1),
    Coord_per_S     = coord_relations / sentences,
    .groups = "drop"
  )
print(coordination_df)
#6-5.Complex Nominal per Clause and/or Sentence
nominal_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    complex_nominals = sum(is_complex_nominal),
    clauses          = sum(is_clause),
    sentences        = n_distinct(sentence_id),
    CN_per_C         = complex_nominals / pmax(clauses, 1),
    CN_per_S         = complex_nominals / sentences,
    .groups = "drop"
  )
print(nominal_df)
#7. Combine all measures into the MLS DF
all_measures <- mls_df %>%
  left_join(clausal_density_df %>% select(document, C_per_S), by = "document") %>%
  left_join(subordination_df %>% select(document, DC_per_C, DC_per_S), by = "document") %>%
  left_join(coordination_df %>% select(document, Coord_per_C, Coord_per_S), by = "document") %>%
  left_join(nominal_df %>% select(document, CN_per_C, CN_per_S), by = "document")
all_measures %>%
  knitr::kable(
    digits = 2,
    col.names = c("Document", "MLS", "C/S", "DC/C", "DC/S", 
                  "Coord/C", "Coord/S", "CN/C", "CN/S")
  )
print(all_measures)
#8. Visualization
syntax_long <- all_measures %>%
  pivot_longer(
    cols = -document,
    names_to = "Measure",
    values_to = "Value"
  ) %>%
  mutate(
    Category = case_when(
      Measure == "MLS" ~ "Sentence Length",
      Measure == "C_per_S" ~ "Clausal Density",
      Measure %in% c("DC_per_C", "DC_per_S") ~ "Subordination",
      Measure %in% c("Coord_per_C", "Coord_per_S") ~ "Coordination",
      Measure %in% c("CN_per_C", "CN_per_S") ~ "Phrasal Complexity"
    )
  )
ggplot(syntax_long, aes(x = Measure, y = Value, fill = document)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_wrap(~Category, scales = "free", ncol = 2) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Syntactic Complexity: Complete Profile",
    subtitle = "Comparing multiple dimensions of syntactic complexity",
    x = NULL,
    y = "Value",
    fill = "Document"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )