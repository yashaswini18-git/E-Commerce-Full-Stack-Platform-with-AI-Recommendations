import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# SHOPSMART AI RECOMMENDATION SYSTEM
# ============================================================

DATA_PATH = os.path.join("data", "train.csv")

MAX_FEATURES = 5000

# Model is loaded only when recommendations are requested
_model_df = None
_vectorizer = None
_tfidf_matrix = None


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================

def decode_text(text):
    """
    Kept for compatibility with app.py.
    The new dataset is already clean, so no decoding is required.
    """
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass

    return str(text)


def extract_clean_name(name, brand=None, category=None):
    """
    The new dataset already contains clean product names.
    Simply return the original name.
    """
    if name is None:
        return ""

    try:
        if pd.isna(name):
            return ""
    except Exception:
        pass

    return str(name).strip()


def looks_corrupted(text):
    """
    Kept for compatibility.
    The replacement dataset contains clean product names.
    """
    if text is None:
        return True

    try:
        if pd.isna(text):
            return True
    except Exception:
        pass

    return not bool(str(text).strip())


def is_raw_name_corrupted(raw_name, cleaned_name):
    """
    Kept for compatibility with the previous application.
    New dataset does not require corruption filtering.
    """
    return False


# ============================================================
# BUILD RECOMMENDATION MODEL
# ============================================================

def _build_model():

    global _model_df
    global _vectorizer
    global _tfidf_matrix

    # Already loaded
    if (
        _model_df is not None
        and _vectorizer is not None
        and _tfidf_matrix is not None
    ):
        return

    print("")
    print("=" * 60)
    print("SHOPSMART AI RECOMMENDATION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load clean dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    print("Products loaded:", len(df))

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "id",
        "name",
        "price"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"train.csv does not contain required column: {column}"
            )

    # --------------------------------------------------------
    # Create optional columns if missing
    # --------------------------------------------------------

    optional_columns = [
        "brand",
        "variant",
        "product_search_description",
        "discounted_price",
        "usage",
        "image_url"
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = ""

    # --------------------------------------------------------
    # Clean missing values
    # --------------------------------------------------------

    text_columns = [
        "name",
        "brand",
        "variant",
        "product_search_description",
        "usage",
        "image_url"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    ).fillna(0)

    df["discounted_price"] = pd.to_numeric(
        df["discounted_price"],
        errors="coerce"
    ).fillna(df["price"])

    # --------------------------------------------------------
    # Clean IDs
    # --------------------------------------------------------

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["id"]
    ).reset_index(drop=True)

    df["id"] = df["id"].astype(int)

    # --------------------------------------------------------
    # Clean product names
    # --------------------------------------------------------

    df["clean_name"] = df["name"].apply(
        extract_clean_name
    )

    # --------------------------------------------------------
    # Create category-like information
    #
    # The new dataset doesn't have a category column.
    # product_search_description contains category information.
    # We keep it as the product description.
    # --------------------------------------------------------

    df["category"] = df["product_search_description"]

    df["description"] = df["product_search_description"]

    # --------------------------------------------------------
    # Create recommendation text
    #
    # Product name is repeated to give it stronger importance.
    # --------------------------------------------------------

    df["combined_text"] = (
        df["clean_name"] + " "
        + df["clean_name"] + " "
        + df["brand"] + " "
        + df["variant"] + " "
        + df["product_search_description"] + " "
        + df["usage"]
    )

    df["combined_text"] = (
        df["combined_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # TF-IDF MODEL
    # --------------------------------------------------------

    print("Creating AI recommendation model...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_text"]
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    _model_df = df
    _vectorizer = vectorizer
    _tfidf_matrix = tfidf_matrix

    print("=" * 60)
    print("AI recommendation model created successfully!")
    print("Number of products:", len(df))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("=" * 60)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity_safe(
    query_vector,
    matrix
):
    """
    Calculate cosine similarity without
    sklearn.metrics.pairwise.cosine_similarity.
    """

    query = query_vector.toarray()

    # Dot product
    scores = matrix @ query.T

    scores = np.asarray(
        scores
    ).reshape(-1)

    # Query norm
    query_norm = np.linalg.norm(
        query
    )

    # Matrix norms
    matrix_norms = np.sqrt(
        matrix.multiply(matrix).sum(
            axis=1
        )
    )

    matrix_norms = np.asarray(
        matrix_norms
    ).reshape(-1)

    denominator = (
        matrix_norms * query_norm
    )

    denominator[
        denominator == 0
    ] = 1e-10

    return scores / denominator


# ============================================================
# GET RECOMMENDATIONS
# ============================================================

def get_recommendations(
    product_id,
    number_of_recommendations=5
):

    # Build model if necessary
    _build_model()

    # --------------------------------------------------------
    # Validate product ID
    # --------------------------------------------------------

    try:
        product_id = int(product_id)
    except (
        ValueError,
        TypeError
    ):
        return pd.DataFrame()

    # --------------------------------------------------------
    # Find product
    # --------------------------------------------------------

    matches = _model_df.index[
        _model_df["id"] == product_id
    ].tolist()

    if not matches:
        return pd.DataFrame()

    product_index = matches[0]

    # --------------------------------------------------------
    # Product vector
    # --------------------------------------------------------

    query_vector = _tfidf_matrix[
        product_index
    ]

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarity_scores = cosine_similarity_safe(
        query_vector,
        _tfidf_matrix
    )

    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    similar_indices = np.argsort(
        similarity_scores
    )[::-1]

    # Remove current product
    similar_indices = [
        index
        for index in similar_indices
        if index != product_index
    ]

    # --------------------------------------------------------
    # Build recommendations
    # --------------------------------------------------------

    recommendations = []

    for index in similar_indices:

        row = _model_df.iloc[index]

        try:
            rec_id = int(row["id"])
        except Exception:
            continue

        rec_name = str(
            row.get(
                "clean_name",
                row.get("name", "")
            )
        ).strip()

        if not rec_name:
            continue

        rec_brand = str(
            row.get(
                "brand",
                ""
            )
        ).strip()

        rec_category = str(
            row.get(
                "category",
                ""
            )
        ).strip()

        rec_description = str(
            row.get(
                "description",
                ""
            )
        ).strip()

        rec_variant = str(
            row.get(
                "variant",
                ""
            )
        ).strip()

        rec_usage = str(
            row.get(
                "usage",
                ""
            )
        ).strip()

        rec_image = str(
            row.get(
                "image_url",
                ""
            )
        ).strip()

        # Price
        try:
            rec_price = float(
                row.get(
                    "price",
                    0
                )
            )
        except Exception:
            rec_price = 0.0

        # Discounted price
        try:
            rec_discounted_price = float(
                row.get(
                    "discounted_price",
                    rec_price
                )
            )
        except Exception:
            rec_discounted_price = rec_price

        similarity = float(
            similarity_scores[index]
        )

        recommendations.append({
            "id": rec_id,
            "name": rec_name,
            "brand": rec_brand,
            "category": rec_category,
            "description": rec_description,
            "variant": rec_variant,
            "usage": rec_usage,
            "price": rec_price,
            "discounted_price": rec_discounted_price,
            "image_url": rec_image,
            "rating": "",
            "similarity_score": similarity
        })

        if len(recommendations) >= number_of_recommendations:
            break

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# MODEL STATUS
# ============================================================

def model_is_loaded():

    return (
        _model_df is not None
        and _vectorizer is not None
        and _tfidf_matrix is not None
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print("ShopSmart AI Recommendation System")
    print("=" * 60)
    print("Dataset:", DATA_PATH)
    print("Recommendation model loads lazily.")
    print("Clean product dataset enabled.")
    print("sklearn.metrics is NOT used.")
    print("=" * 60)