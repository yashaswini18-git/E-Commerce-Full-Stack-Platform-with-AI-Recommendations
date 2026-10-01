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
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

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

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///shopsmart.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# DATABASE MODELS
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

    print("=" * 60)
    print("Product dataset loaded successfully!")
    print(f"Number of products: {len(df)}")
    print("Dataset columns:")
    print(list(df.columns))
    print("=" * 60)

except Exception as e:

    print(
        "Error loading product dataset:",
        e
    )

    df = pd.DataFrame()


# ============================================================
# DATASET COMPATIBILITY
# ============================================================

OPTIONAL_COLUMNS = [
    "product_search_description",
    "variant",
    "brand",
    "price",
    "discounted_price",
    "usage",
    "image_url"
]

for column in OPTIONAL_COLUMNS:

    if column not in df.columns:
        df[column] = ""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    try:

        if pd.isna(value):
            return 0.0

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return 0.0


def safe_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


# ============================================================
# GET PRODUCT BY ID
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

    if "id" not in df.columns:
        return None

    matches = df[
        df["id"] == product_id
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    name = safe_text(
        row.get(
            "name",
            ""
        )
    )

    brand = safe_text(
        row.get(
            "brand",
            ""
        )
    )

    description = safe_text(
        row.get(
            "product_search_description",
            ""
        )
    )

    variant = safe_text(
        row.get(
            "variant",
            ""
        )
    )

    usage = safe_text(
        row.get(
            "usage",
            ""
        )
    )

    image_url = safe_text(
        row.get(
            "image_url",
            ""
        )
    )

    price = safe_float(
        row.get(
            "price",
            0
        )
    )

    discounted_price = safe_float(
        row.get(
            "discounted_price",
            0
        )
    )

    # New dataset does not have a separate category column.
    # We use product_search_description as category information.
    category = description

    rating = "No rating available"

    return {

        "id": product_id,

        "name": name,

        "raw_name": name,

        "category": category,

        "price": price,

        "brand": brand,

        "rating": rating,

        "description": description,

        "product_specification": variant,

        "variant": variant,

        "usage": usage,

        "discounted_price": discounted_price,

        "image_url": image_url
    }


# ============================================================
# PRODUCT FROM DATAFRAME ROW
# ============================================================

def product_from_row(row):

    try:

        product_id = int(
            row["id"]
        )

    except (
        ValueError,
        TypeError,
        KeyError
    ):

        return None

    return get_product_by_id(
        product_id
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    featured_products = []

    if not df.empty:

        for _, row in df.head(12).iterrows():

            product = product_from_row(
                row
            )

            if product:

                featured_products.append(
                    product
                )

    return render_template(
        "index.html",
        products=featured_products
    )


# ============================================================
# PRODUCTS
# ============================================================

@app.route("/products")
def products():

    products_list = []

    search = request.args.get(
        "search",
        ""
    ).strip().lower()

    if not df.empty:

        filtered_df = df

        if search:

            name_mask = pd.Series(
                False,
                index=df.index
            )

            brand_mask = pd.Series(
                False,
                index=df.index
            )

            description_mask = pd.Series(
                False,
                index=df.index
            )

            if "name" in df.columns:

                name_mask = (
                    df["name"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search,
                        regex=False
                    )
                )

            if "brand" in df.columns:

                brand_mask = (
                    df["brand"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search,
                        regex=False
                    )
                )

            if "product_search_description" in df.columns:

                description_mask = (
                    df[
                        "product_search_description"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search,
                        regex=False
                    )
                )

            filtered_df = df[
                name_mask
                | brand_mask
                | description_mask
            ]

        for _, row in filtered_df.head(100).iterrows():

            product = product_from_row(
                row
            )

            if product:

                products_list.append(
                    product
                )

    return render_template(
        "products.html",
        products=products_list,
        search=search
    )


# ============================================================
# PRODUCT DETAILS
# ============================================================

@app.route(
    "/product/<int:product_id>"
)
def product(product_id):

    product_data = get_product_by_id(
        product_id
    )

    if not product_data:

        flash(
            "Product not found.",
            "danger"
        )

        return redirect(
            url_for("products")
        )

    similar_products = []

    try:

        recommendations_data = get_recommendations(
            product_id,
            5
        )

        if isinstance(
            recommendations_data,
            pd.DataFrame
        ):

            recommendations_data = (
                recommendations_data.to_dict(
                    orient="records"
                )
            )

        for item in recommendations_data:

            if not isinstance(
                item,
                dict
            ):
                continue

            rec_id = item.get(
                "id"
            )

            if rec_id is None:
                continue

            rec_product = get_product_by_id(
                rec_id
            )

            if rec_product:

                rec_product[
                    "similarity_score"
                ] = item.get(
                    "similarity_score",
                    0
                )

                similar_products.append(
                    rec_product
                )

    except Exception as e:

        print(
            "Product recommendation error:",
            e
        )

    return render_template(
        "product.html",
        product=product_data,
        recommendations=similar_products,
        similar_products=similar_products
    )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

@app.route("/recommendations")
def recommendations():

    product_id = request.args.get(
        "product_id"
    )

    recommended_products = []

    # --------------------------------------------------------
    # SELECTED PRODUCT RECOMMENDATIONS
    # --------------------------------------------------------

    if product_id:

        try:

            product_id = int(
                product_id
            )

            recommendations_data = get_recommendations(
                product_id,
                20
            )

            if isinstance(
                recommendations_data,
                pd.DataFrame
            ):

                recommendations_data = (
                    recommendations_data.to_dict(
                        orient="records"
                    )
                )

            for item in recommendations_data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                rec_id = item.get(
                    "id"
                )

                if rec_id is None:
                    continue

                rec_product = get_product_by_id(
                    rec_id
                )

                if rec_product:

                    rec_product[
                        "similarity_score"
                    ] = item.get(
                        "similarity_score",
                        0
                    )

                    recommended_products.append(
                        rec_product
                    )

        except Exception as e:

            print(
                "Recommendation error:",
                e
            )

    # --------------------------------------------------------
    # GENERAL AI PICKS
    # --------------------------------------------------------

    if (
        not recommended_products
        and not df.empty
    ):

        try:

            first_product_id = int(
                df.iloc[0]["id"]
            )

            recommendations_data = get_recommendations(
                first_product_id,
                20
            )

            if isinstance(
                recommendations_data,
                pd.DataFrame
            ):

                recommendations_data = (
                    recommendations_data.to_dict(
                        orient="records"
                    )
                )

            for item in recommendations_data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                rec_id = item.get(
                    "id"
                )

                if rec_id is None:
                    continue

                rec_product = get_product_by_id(
                    rec_id
                )

                if rec_product:

                    rec_product[
                        "similarity_score"
                    ] = item.get(
                        "similarity_score",
                        0
                    )

                    recommended_products.append(
                        rec_product
                    )

        except Exception as e:

            print(
                "General AI recommendation error:",
                e
            )

    return render_template(
        "recommendations.html",
        recommendations=recommended_products,
        products=recommended_products
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
                "warning"
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

        db.session.add(
            user
        )

        db.session.commit()

        flash(
            "Account created successfully. Please login.",
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
# ADD TO CART
# ============================================================

@app.route(
    "/add-to-cart/<int:product_id>",
    methods=["POST", "GET"]
)
def add_to_cart(product_id):

    product_data = get_product_by_id(
        product_id
    )

    if not product_data:

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

    if not isinstance(
        cart,
        dict
    ):

        cart = {}

    key = str(
        product_id
    )

    # --------------------------------------------------------
    # PRODUCT ALREADY EXISTS
    # --------------------------------------------------------

    if key in cart:

        cart[key]["quantity"] = (
            int(
                cart[key].get(
                    "quantity",
                    1
                )
            ) + 1
        )

        cart[key]["image_url"] = product_data.get(
            "image_url",
            ""
        )

        cart[key]["brand"] = product_data.get(
            "brand",
            ""
        )

        cart[key]["category"] = product_data.get(
            "category",
            ""
        )

        cart[key]["variant"] = product_data.get(
            "variant",
            ""
        )

    # --------------------------------------------------------
    # NEW PRODUCT
    # --------------------------------------------------------

    else:

        cart[key] = {

            "id": product_data["id"],

            "name": product_data["name"],

            "brand": product_data.get(
                "brand",
                ""
            ),

            "category": product_data.get(
                "category",
                ""
            ),

            "description": product_data.get(
                "description",
                ""
            ),

            "variant": product_data.get(
                "variant",
                ""
            ),

            "usage": product_data.get(
                "usage",
                ""
            ),

            "image_url": product_data.get(
                "image_url",
                ""
            ),

            "price": product_data["price"],

            "discounted_price": product_data.get(
                "discounted_price",
                0
            ),

            "quantity": 1
        }

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

    cart = session.get(
        "cart",
        {}
    )

    if not isinstance(
        cart,
        dict
    ):

        cart = {}

    cart_items = []

    total = 0

    for key, item in cart.items():

        try:

            product_id = int(
                item.get(
                    "id",
                    key
                )
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        # Always get fresh product information
        # from train.csv.
        product_data = get_product_by_id(
            product_id
        )

        if not product_data:
            continue

        try:

            quantity = int(
                item.get(
                    "quantity",
                    1
                )
            )

        except (
            ValueError,
            TypeError
        ):

            quantity = 1

        if quantity < 1:
            quantity = 1

        original_price = safe_float(
            product_data.get(
                "price",
                0
            )
        )

        discounted_price = safe_float(
            product_data.get(
                "discounted_price",
                0
            )
        )

        if (
            discounted_price > 0
            and discounted_price < original_price
        ):

            price = discounted_price

        else:

            price = original_price

        subtotal = (
            price * quantity
        )

        cart_item = {

            "id": product_id,

            "name": product_data.get(
                "name",
                ""
            ),

            "brand": product_data.get(
                "brand",
                ""
            ),

            "category": product_data.get(
                "category",
                ""
            ),

            "description": product_data.get(
                "description",
                ""
            ),

            "variant": product_data.get(
                "variant",
                ""
            ),

            "usage": product_data.get(
                "usage",
                ""
            ),

            "image_url": product_data.get(
                "image_url",
                ""
            ),

            "price": price,

            "original_price": original_price,

            "discounted_price": discounted_price,

            "quantity": quantity,

            "subtotal": subtotal
        }

        cart_items.append(
            cart_item
        )

        total += subtotal

    session["cart"] = cart

    session.modified = True

    return render_template(
        "cart.html",
        cart_items=cart_items,
        cart_products=cart_items,
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

    if not isinstance(
        cart,
        dict
    ):

        cart = {}

    key = str(
        product_id
    )

    if key in cart:

        del cart[key]

        flash(
            "Product removed from cart.",
            "success"
        )

    session["cart"] = cart

    session.modified = True

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

    cart = session.get(
        "cart",
        {}
    )

    if not cart:

        flash(
            "Your cart is empty.",
            "warning"
        )

        return redirect(
            url_for("products")
        )

    cart_items = []

    total = 0

    # --------------------------------------------------------
    # BUILD CHECKOUT ITEMS
    # --------------------------------------------------------

    for key, item in cart.items():

        try:

            product_id = int(
                item.get(
                    "id",
                    key
                )
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        # Get complete product information.
        product_data = get_product_by_id(
            product_id
        )

        if not product_data:
            continue

        try:

            quantity = int(
                item.get(
                    "quantity",
                    1
                )
            )

        except (
            ValueError,
            TypeError
        ):

            quantity = 1

        if quantity < 1:
            quantity = 1

        original_price = safe_float(
            product_data.get(
                "price",
                0
            )
        )

        discounted_price = safe_float(
            product_data.get(
                "discounted_price",
                0
            )
        )

        if (
            discounted_price > 0
            and discounted_price < original_price
        ):

            price = discounted_price

        else:

            price = original_price

        subtotal = (
            price * quantity
        )

        total += subtotal

        # ----------------------------------------------------
        # IMPORTANT:
        # Include category because checkout.html uses it.
        # ----------------------------------------------------

        cart_items.append({

            "id": product_data["id"],

            "name": product_data["name"],

            "brand": product_data.get(
                "brand",
                ""
            ),

            "category": product_data.get(
                "category",
                ""
            ),

            "description": product_data.get(
                "description",
                ""
            ),

            "variant": product_data.get(
                "variant",
                ""
            ),

            "usage": product_data.get(
                "usage",
                ""
            ),

            "image_url": product_data.get(
                "image_url",
                ""
            ),

            "price": price,

            "original_price": original_price,

            "discounted_price": discounted_price,

            "quantity": quantity,

            "subtotal": subtotal
        })

    return render_template(
        "checkout.html",
        cart_items=cart_items,
        cart_products=cart_items,
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

    cart = session.get(
        "cart",
        {}
    )

    if not cart:

        flash(
            "Your cart is empty.",
            "warning"
        )

        return redirect(
            url_for("products")
        )

    total = 0

    order_items_data = []

    # --------------------------------------------------------
    # CALCULATE TOTAL
    # --------------------------------------------------------

    for key, item in cart.items():

        try:

            product_id = int(
                item.get(
                    "id",
                    key
                )
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        product_data = get_product_by_id(
            product_id
        )

        if not product_data:
            continue

        try:

            quantity = int(
                item.get(
                    "quantity",
                    1
                )
            )

        except (
            ValueError,
            TypeError
        ):

            quantity = 1

        if quantity < 1:
            quantity = 1

        original_price = safe_float(
            product_data.get(
                "price",
                0
            )
        )

        discounted_price = safe_float(
            product_data.get(
                "discounted_price",
                0
            )
        )

        if (
            discounted_price > 0
            and discounted_price < original_price
        ):

            price = discounted_price

        else:

            price = original_price

        subtotal = (
            price * quantity
        )

        total += subtotal

        order_items_data.append({

            "product_id": product_data["id"],

            "quantity": quantity,

            "price": price
        })

    if not order_items_data:

        flash(
            "No valid products found in cart.",
            "danger"
        )

        return redirect(
            url_for("cart")
        )

    # --------------------------------------------------------
    # CREATE ORDER
    # --------------------------------------------------------

    order = Order(

        user_id=session["user_id"],

        total_amount=total,

        status="Placed"
    )

    db.session.add(
        order
    )

    db.session.flush()

    # --------------------------------------------------------
    # CREATE ORDER ITEMS
    # --------------------------------------------------------

    for item in order_items_data:

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

    # --------------------------------------------------------
    # CLEAR CART AFTER ORDER
    # --------------------------------------------------------

    session["cart"] = {}

    session.modified = True

    return redirect(
        url_for(
            "order_success",
            order_id=order.id
        )
    )


# ============================================================
# ORDER SUCCESS
# ============================================================

@app.route(
    "/order-success/<int:order_id>"
)
def order_success(order_id):

    order = Order.query.get(
        order_id
    )

    if not order:

        flash(
            "Order not found.",
            "danger"
        )

        return redirect(
            url_for("home")
        )

    return render_template(
        "order_success.html",
        order=order
    )


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

with app.app_context():

    try:

        db.create_all()

        print(
            "Database tables ready."
        )

    except Exception as e:

        print(
            "Database initialization warning:",
            e
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Starting ShopSmart...")
    print("AI-powered e-commerce platform")
    print(f"Products loaded: {len(df)}")
    print("Cart image support: ENABLED")
    print("Checkout category support: ENABLED")
    print("=" * 60)

    app.run(
        debug=True
    )