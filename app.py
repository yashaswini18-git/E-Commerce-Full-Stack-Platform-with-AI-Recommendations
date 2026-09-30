from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import os

from dotenv import load_dotenv

from ai.recommendation import get_recommendations


# ============================================================
# APP CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "shopsmart-secret-key"
)

app.config["DATABASE_URL"] = os.getenv(
    "DATABASE_URL"
)

# Use SQLite if DATABASE_URL is not available
database_url = os.getenv("DATABASE_URL")

if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///shopsmart.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# ============================================================
# DATABASE
# ============================================================

db = SQLAlchemy(app)


# ============================================================
# USER MODEL
# ============================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )


# ============================================================
# ORDER MODEL
# ============================================================

class Order(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    total_amount = db.Column(
        db.Float,
        nullable=False
    )

    status = db.Column(
        db.String(50),
        default="Placed"
    )


# ============================================================
# ORDER ITEM MODEL
# ============================================================

class OrderItem(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    order_id = db.Column(
        db.Integer,
        db.ForeignKey("order.id"),
        nullable=False
    )

    product_id = db.Column(
        db.Integer,
        nullable=False
    )

    quantity = db.Column(
        db.Integer,
        default=1
    )

    price = db.Column(
        db.Float,
        nullable=False
    )


# ============================================================
# LOAD PRODUCT DATASET
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "train.csv"
)

try:

    df = pd.read_csv(DATA_PATH)

    print(
        f"Product dataset loaded successfully!"
    )

    print(
        f"Number of products: {len(df)}"
    )

except Exception as e:

    print(
        "Error loading product dataset:",
        e
    )

    df = pd.DataFrame()


# ============================================================
# CLEAN PRODUCT NAME
# ============================================================

def clean_product_name(
    name,
    brand=None,
    category=None
):

    if pd.isna(name):

        name = ""

    name = str(name).strip()

    if name:

        return name

    if brand and not pd.isna(brand):

        return str(brand)

    if category and not pd.isna(category):

        return str(category)

    return "Product"


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_product_by_id(product_id):

    if df.empty:

        return None

    try:

        product_id = int(product_id)

    except (
        ValueError,
        TypeError
    ):

        return None

    product_data = df[
        df["id"] == product_id
    ]

    if product_data.empty:

        return None

    row = product_data.iloc[0]

    return {
        "id": int(row["id"]),

        "name": clean_product_name(
            row["name"],
            row["brand"],
            row["category"]
        ),

        "price": float(row["price"]),

        "category": str(
            row["category"]
        ),

        "brand": str(
            row["brand"]
        )
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    products = []

    if not df.empty:

        for _, row in df.head(12).iterrows():

            products.append({

                "id": int(row["id"]),

                "name": clean_product_name(
                    row["name"],
                    row["brand"],
                    row["category"]
                ),

                "price": float(row["price"]),

                "category": str(
                    row["category"]
                ),

                "brand": str(
                    row["brand"]
                )
            })

    return render_template(
        "index.html",
        products=products
    )


# ============================================================
# PRODUCTS PAGE
# ============================================================

@app.route("/products")
def products():

    search = request.args.get(
        "search",
        ""
    ).strip()

    category = request.args.get(
        "category",
        ""
    ).strip()

    products = []

    if not df.empty:

        filtered_df = df.copy()

        # Search
        if search:

            search_lower = search.lower()

            filtered_df = filtered_df[
                filtered_df.apply(
                    lambda row:
                    search_lower in str(
                        row.get("name", "")
                    ).lower()
                    or
                    search_lower in str(
                        row.get("brand", "")
                    ).lower()
                    or
                    search_lower in str(
                        row.get("category", "")
                    ).lower(),
                    axis=1
                )
            ]

        # Category
        if category:

            filtered_df = filtered_df[
                filtered_df["category"]
                .astype(str)
                .str.lower()
                ==
                category.lower()
            ]

        for _, row in filtered_df.head(100).iterrows():

            products.append({

                "id": int(row["id"]),

                "name": clean_product_name(
                    row["name"],
                    row["brand"],
                    row["category"]
                ),

                "price": float(row["price"]),

                "category": str(
                    row["category"]
                ),

                "brand": str(
                    row["brand"]
                )
            })

    return render_template(
        "products.html",
        products=products,
        search=search,
        category=category
    )


# ============================================================
# PRODUCT DETAILS
# ============================================================

@app.route("/product/<int:product_id>")
def product(product_id):

    product = get_product_by_id(
        product_id
    )

    if product is None:

        flash(
            "Product not found.",
            "danger"
        )

        return redirect(
            url_for("products")
        )

    # AI recommendations
    recommendations = []

    try:

        recommended_ids = get_recommendations(
            product_id,
            number_of_recommendations=5
        )

        for recommended_id in recommended_ids:

            try:

                recommended_id = int(
                    recommended_id
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            recommended_product = (
                get_product_by_id(
                    recommended_id
                )
            )

            if recommended_product:

                recommendations.append(
                    recommended_product
                )

    except Exception as e:

        print(
            "Recommendation error:",
            e
        )

    return render_template(
        "product.html",
        product=product,
        recommendations=recommendations
    )


# ============================================================
# AI RECOMMENDATIONS PAGE
# ============================================================

@app.route("/recommendations")
def recommendations():

    product_id = request.args.get(
        "product_id",
        type=int
    )

    # Use first product if no product is selected
    if product_id is None:

        if df.empty:

            return render_template(
                "recommendations.html",
                recommendations=[]
            )

        product_id = int(
            df.iloc[0]["id"]
        )

    # Get AI recommendations
    try:

        recommended_ids = get_recommendations(
            product_id,
            number_of_recommendations=5
        )

    except Exception as e:

        print(
            "AI recommendation error:",
            e
        )

        recommended_ids = []

    # Convert recommendation IDs
    # into product dictionaries

    recommended_products = []

    for recommended_id in recommended_ids:

        try:

            recommended_id = int(
                recommended_id
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        product_data = df[
            df["id"] == recommended_id
        ]

        if product_data.empty:

            continue

        row = product_data.iloc[0]

        recommended_products.append({

            "id": int(
                row["id"]
            ),

            "name": clean_product_name(
                row["name"],
                row["brand"],
                row["category"]
            ),

            "price": float(
                row["price"]
            ),

            "category": str(
                row["category"]
            ),

            "brand": str(
                row["brand"]
            )

        })

    return render_template(
        "recommendations.html",
        recommendations=recommended_products
    )


# ============================================================
# ADD TO CART
# ============================================================

@app.route("/add-to-cart/<int:product_id>")
def add_to_cart(product_id):

    product = get_product_by_id(
        product_id
    )

    if product is None:

        flash(
            "Product not found.",
            "danger"
        )

        return redirect(
            url_for("products")
        )

    cart = session.get(
        "cart",
        {}
    )

    product_id_str = str(
        product_id
    )

    if product_id_str in cart:

        cart[product_id_str] += 1

    else:

        cart[product_id_str] = 1

    session["cart"] = cart

    session.modified = True

    flash(
        "Product added to cart!",
        "success"
    )

    return redirect(
        request.referrer
        or url_for("products")
    )


# ============================================================
# CART
# ============================================================

@app.route("/cart")
def cart():

    cart_data = session.get(
        "cart",
        {}
    )

    cart_products = []

    total = 0

    for product_id, quantity in cart_data.items():

        product = get_product_by_id(
            product_id
        )

        if product is None:

            continue

        product["quantity"] = quantity

        product["subtotal"] = (
            product["price"] * quantity
        )

        total += product["subtotal"]

        cart_products.append(
            product
        )

    return render_template(
        "cart.html",
        cart_products=cart_products,
        total=total
    )


# ============================================================
# REMOVE FROM CART
# ============================================================

@app.route(
    "/remove-from-cart/<int:product_id>"
)
def remove_from_cart(product_id):

    cart = session.get(
        "cart",
        {}
    )

    product_id_str = str(
        product_id
    )

    if product_id_str in cart:

        del cart[
            product_id_str
        ]

    session["cart"] = cart

    session.modified = True

    flash(
        "Product removed from cart.",
        "success"
    )

    return redirect(
        url_for("cart")
    )


# ============================================================
# CLEAR CART
# ============================================================

@app.route("/clear-cart")
def clear_cart():

    session["cart"] = {}

    session.modified = True

    flash(
        "Cart cleared.",
        "success"
    )

    return redirect(
        url_for("cart")
    )


# ============================================================
# CHECKOUT
# ============================================================

@app.route("/checkout")
def checkout():

    if "user_id" not in session:

        flash(
            "Please login before checkout.",
            "warning"
        )

        return redirect(
            url_for("login")
        )

    cart_data = session.get(
        "cart",
        {}
    )

    if not cart_data:

        flash(
            "Your cart is empty.",
            "warning"
        )

        return redirect(
            url_for("cart")
        )

    cart_products = []

    total = 0

    for product_id, quantity in cart_data.items():

        product = get_product_by_id(
            product_id
        )

        if product is None:

            continue

        product["quantity"] = quantity

        product["subtotal"] = (
            product["price"] * quantity
        )

        total += product["subtotal"]

        cart_products.append(
            product
        )

    return render_template(
        "checkout.html",
        cart_products=cart_products,
        total=total
    )


# ============================================================
# PLACE ORDER
# ============================================================

@app.route(
    "/place-order",
    methods=["POST"]
)
def place_order():

    if "user_id" not in session:

        flash(
            "Please login before placing an order.",
            "warning"
        )

        return redirect(
            url_for("login")
        )

    cart_data = session.get(
        "cart",
        {}
    )

    if not cart_data:

        flash(
            "Your cart is empty.",
            "warning"
        )

        return redirect(
            url_for("cart")
        )

    total = 0

    order_items = []

    for product_id, quantity in cart_data.items():

        product = get_product_by_id(
            product_id
        )

        if product is None:

            continue

        subtotal = (
            product["price"] * quantity
        )

        total += subtotal

        order_items.append({

            "product_id": product["id"],

            "quantity": quantity,

            "price": product["price"]

        })

    # Create order
    order = Order(

        user_id=session["user_id"],

        total_amount=total,

        status="Placed"

    )

    db.session.add(order)

    db.session.flush()

    # Add order items
    for item in order_items:

        order_item = OrderItem(

            order_id=order.id,

            product_id=item["product_id"],

            quantity=item["quantity"],

            price=item["price"]

        )

        db.session.add(
            order_item
        )

    db.session.commit()

    # Empty cart
    session["cart"] = {}

    session.modified = True

    flash(
        "Order placed successfully!",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "danger"
            )

            return redirect(
                url_for("signup")
            )

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "Email already registered.",
                "danger"
            )

            return redirect(
                url_for("signup")
            )

        hashed_password = generate_password_hash(
            password
        )

        user = User(

            name=name,

            email=email,

            password=hashed_password

        )

        db.session.add(user)

        db.session.commit()

        flash(
            "Account created successfully! Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "signup.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user_id"] = user.id

            session["user_name"] = user.name

            flash(
                "Login successful!",
                "success"
            )

            return redirect(
                url_for("home")
            )

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.pop(
        "user_id",
        None
    )

    session.pop(
        "user_name",
        None
    )

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "AI recommendation model loaded."
    )

    print(
        "Starting ShopSmart..."
    )

    app.run(
        debug=True
    )