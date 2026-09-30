import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Decode only the encoded product name/text
# ============================================================

def decode_text(text):
    if not isinstance(text, str):
        return text

    result = ""

    for char in text:

        if 'a' <= char <= 'z':
            result += chr(
                (ord(char) - ord('a') - 3) % 26 + ord('a')
            )

        elif 'A' <= char <= 'Z':
            result += chr(
                (ord(char) - ord('A') - 3) % 26 + ord('A')
            )

        else:
            result += char

    return result


# ============================================================
# Load dataset
# ============================================================

DATA_PATH = "data/train.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# Fill missing values
# ============================================================

text_columns = [
    "name",
    "category",
    "brand",
    "description",
    "specs"
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str)


# ============================================================
# Decode PRODUCT NAME only
# ============================================================

df["name"] = df["name"].apply(decode_text)


# ============================================================
# Create combined product information
# ============================================================

df["combined_text"] = (
    df["name"] + " "
    + df["category"] + " "
    + df["brand"] + " "
    + df["description"].str[:1000] + " "
    + df["specs"].str[:1000]
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    df["combined_text"]
)


print("AI recommendation model created successfully!")
print("Number of products:", len(df))
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# Recommendation Function
# ============================================================

def get_recommendations(
    product_id,
    number_of_recommendations=5
):

    matches = df.index[
        df["id"] == product_id
    ].tolist()

    if not matches:
        return pd.DataFrame()

    product_index = matches[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[product_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    # Remove current product
    similar_indices = [
        index
        for index in similar_indices
        if index != product_index
    ]

    recommended_indices = similar_indices[
        :number_of_recommendations
    ]

    recommendations = df.iloc[
        recommended_indices
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[index]
        for index in recommended_indices
    ]

    return recommendations[
        [
            "id",
            "name",
            "category",
            "price",
            "rating",
            "brand",
            "similarity_score"
        ]
    ]


# ============================================================
# Test AI Recommendation
# ============================================================

if __name__ == "__main__":

    first_product_id = df.iloc[0]["id"]

    print("\nSelected product:")

    print(
        df.iloc[0][
            [
                "id",
                "name",
                "category",
                "brand"
            ]
        ]
    )

    print("\nAI Recommendations:")

    recommendations = get_recommendations(
        first_product_id,
        5
    )

    print(
        recommendations.to_string(
            index=False
        )
    )