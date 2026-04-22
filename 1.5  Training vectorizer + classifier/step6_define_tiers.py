# Defining them according to the results of my word2vec
# Tier definitions
# Tier A: direct spellings / variants of the seed concept
TIER_A = {
    "queer", "queers","酷儿"
}

# Tier B: closely related queer terms
TIER_B = {
    "queer theory","酷儿理论","酷儿性","queerness"
}

# Tier C: "maybe" neighborhood (often adjacent, not always queer-specific)
# I picked some words(not all) from word2vec model
TIER_C = {
    "gender", "性别",
    "woman", "women", "女性", "男性",
    "性少数","lgbtq",
    "巴特勒", "theory",
    "gay", "同性恋", "le", "lesbian", "trans","亚文化", "交叉性",
    "feminism", "女性主义", "女权主义","身份认同","酷酷","边缘",
    "intersectionality", "交叉性","异性恋"
}

print("Tier A size:", len(TIER_A))
print("Tier B size:", len(TIER_B))
print("Tier C size:", len(TIER_C))

print("\nTier A:", sorted(TIER_A))
print("\nTier B:", sorted(TIER_B))
print("\nTier C:", sorted(TIER_C))