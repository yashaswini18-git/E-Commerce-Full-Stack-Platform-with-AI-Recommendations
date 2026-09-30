from flask import Flask, render_template, request, redirect, url_for, session, flash
from ai.recommendation import df, get_recommendations
import re


app = Flask(__name__)

# Secret key for Flask sessions
app.config["SECRET_KEY"] = "ecommerce-secret-key"


# =========================================================
# CLEAN PRODUCT NAME
# =========================================================

def clean_product_name(name, brand="", category=""):

    brand = str(brand).strip()
    category = str(category).strip()

    # Women's shapewear
    if "shapewear" in category.lower():

        if brand and brand.lower() != "nan":

            brand = re.sub(
                r"[^A-Za-z0-9 &.-]",
                "",
                brand
            ).strip()

            if brand:
                return f"{brand.title()} Women's Shapewear"

        return "Women's Shapewear"

    # Use brand when available
    if brand and brand.lower() != "nan":

        brand = re.sub(
            r"[^A-Za-z0-9 &.-]",
            "",
            brand
        ).strip()

        if brand:
            return f"{brand.title()} Product"

    # Default name
    return "ShopSmart Product"


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    # Get first 12 products
    products = df.head(12).copy()

    # Get AI recommendations
    first_product_id = df.iloc[0]["id"]

    recommendations = get_recommendations(
        first_product_id,
        6
    ).copy()

    # Clean product names
    products["name"] = products.apply(
        lambda row: clean_product_name(
            row["name"],
            row["brand"],
            row["category"]
        ),
        axis=1
    )

    recommendations["name"] = recommendations.apply(
        lambda row: clean_product_name(
            row["name"],
            row["brand"],
            row["category"]
        ),
        axis=1
    )

    return render_template(
        "index.html",
        products=products.to_dict("records"),
        recommendations=recommendations.to_dict("records")
    )


# =========================================================
# PRODUCTS PAGE
# =========================================================

@app.route("/products")
def products():

    search = request.args.get(
        "search",
        ""
    ).strip()

    # Search products
    if search:

        filtered = df[
            df["name"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            df["category"].str.contains(
                search,
                case=False,
                na=False
            )
            |
            df["brand"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered = df

    # Show maximum 50 products
    products_list = filtered.head(50).copy()

    # Clean product names
    products_list["name"] = products_list.apply(
        lambda row: clean_product_name(
            row["name"],
            row["brand"],
            row["category"]
        ),
        axis=1
    )

    return render_template(
        "products.html",
        products=products_list.to_dict("records"),
        search=search
    )


# =========================================================
# PRODUCT DETAILS
# =========================================================

@app.route("/product/<int:product_id>")
def product_details(product_id):

    # Find product
    product_matches = df[
        df["id"] == product_id
    ]

    if product_matches.empty:

        return "Product not found", 404

    # Get product
    product = product_matches.iloc[0].copy()

    # Clean product name
    product["name"] = clean_product_name(
        product["name"],
        product["brand"],
        product["category"]
    )

    # Get AI recommendations
    recommendations = get_recommendations(
        product_id,
        6
    ).copy()

    # Clean recommendation names
    recommendations["name"] = recommendations.apply(
        lambda row: clean_product_name(
            row["name"],
            row["brand"],
            row["category"]
        ),
        axis=1
    )

    return render_template(
        "product.html",
        product=product.to_dict(),
        recommendations=recommendations.to_dict("records")
    )


# =========================================================
# ADD TO CART
# =========================================================

@app.route("/add-to-cart/<int:product_id>")
def add_to_cart(product_id):

    # Create cart if it doesn't exist
    if "cart" not in session:

        session["cart"] = []

    cart = session["cart"]

    # Add product only if not already present
    if product_id not in cart:

        cart.append(product_id)

    session["cart"] = cart

    flash(
        "Product added to cart successfully!"
    )

    # Return to previous page
    return redirect(
        request.referrer or url_for("products")
    )


# =========================================================
# CART PAGE
# =========================================================

@app.route("/cart")
def cart():

    cart_ids = session.get(
        "cart",
        []
    )

    cart_products = []

    # Get products from dataset
    for product_id in cart_ids:

        matches = df[
            df["id"] == product_id
        ]

        if not matches.empty:

            product = matches.iloc[0].to_dict()

            # Clean product name
            product["name"] = clean_product_name(
                product["name"],
                product["brand"],
                product["category"]
            )

            cart_products.append(product)

    # Calculate total
    total = sum(
        float(product["price"])
        for product in cart_products
    )

    return render_template(
        "cart.html",
        products=cart_products,
        total=total
    )


# =========================================================
# REMOVE PRODUCT FROM CART
# =========================================================

@app.route("/remove-from-cart/<int:product_id>")
def remove_from_cart(product_id):

    cart = session.get(
        "cart",
        []
    )

    if product_id in cart:

        cart.remove(product_id)

    session["cart"] = cart

    return redirect(
        url_for("cart")
    )


# =========================================================
# CLEAR CART
# =========================================================

@app.route("/clear-cart")
def clear_cart():

    session["cart"] = []

    return redirect(
        url_for("cart")
    )


# =========================================================
# CHECKOUT PAGE
# =========================================================

@app.route("/checkout")
def checkout():

    cart_ids = session.get(
        "cart",
        []
    )

    # If cart is empty
    if not cart_ids:

        return redirect(
            url_for("cart")
        )

    cart_products = []

    # Get cart products
    for product_id in cart_ids:

        matches = df[
            df["id"] == product_id
        ]

        if not matches.empty:

            product = matches.iloc[0].to_dict()

            # Clean name
            product["name"] = clean_product_name(
                product["name"],
                product["brand"],
                product["category"]
            )

            cart_products.append(product)

    # Calculate total
    total = sum(
        float(product["price"])
        for product in cart_products
    )

    return render_template(
        "checkout.html",
        products=cart_products,
        total=total
    )


# =========================================================
# PLACE ORDER
# =========================================================

@app.route(
    "/place-order",
    methods=["POST"]
)
def place_order():

    # Get customer details
    name = request.form.get(
        "name"
    )

    email = request.form.get(
        "email"
    )

    phone = request.form.get(
        "phone"
    )

    address = request.form.get(
        "address"
    )

    payment = request.form.get(
        "payment"
    )

    # Get cart
    cart_ids = session.get(
        "cart",
        []
    )

    # If cart is empty
    if not cart_ids:

        return redirect(
            url_for("cart")
        )

    cart_products = []

    # Get products
    for product_id in cart_ids:

        matches = df[
            df["id"] == product_id
        ]

        if not matches.empty:

            product = matches.iloc[0].to_dict()

            # Clean product name
            product["name"] = clean_product_name(
                product["name"],
                product["brand"],
                product["category"]
            )

            cart_products.append(product)

    # Calculate total
    total = sum(
        float(product["price"])
        for product in cart_products
    )

    # Clear cart after placing order
    session["cart"] = []

    # Show order success page
    return render_template(
        "order_success.html",
        name=name,
        email=email,
        phone=phone,
        address=address,
        payment=payment,
        products=cart_products,
        total=total
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )