# ShopSmart - E-Commerce Full-Stack Platform with AI Recommendations

## Project Overview

ShopSmart is a full-stack e-commerce web application developed using Python and Flask. The platform allows users to browse products, view product details, add products to a shopping cart, proceed to checkout, and place orders.

The project also includes an AI-based product recommendation system that recommends similar products based on product information.

## Objectives

- Develop a functional e-commerce web application.
- Provide product browsing and search functionality.
- Implement user registration and login.
- Implement shopping cart and checkout functionality.
- Store orders using a database.
- Provide AI-based product recommendations.
- Create a simple and user-friendly web interface.

## Main Features

### 1. Product Browsing
Users can browse available products and search for products using product names, brands, and product descriptions.

### 2. Product Details
Each product page displays relevant product information such as:

- Product name
- Brand
- Price
- Discounted price
- Product description
- Variant
- Usage information
- Product image

### 3. User Authentication
The application provides:

- User registration
- User login
- Password hashing
- Logout functionality

### 4. Shopping Cart
Users can:

- Add products to the cart
- Increase product quantity
- Remove products
- Clear the cart
- View the total amount

### 5. Checkout and Orders
Users can review their cart during checkout and place an order.

Order information is stored in the SQLite database.

### 6. AI Product Recommendations

ShopSmart uses a content-based recommendation approach.

The recommendation system uses:

- TF-IDF vectorization
- Product names
- Brand information
- Product descriptions
- Variants
- Usage information
- NumPy-based cosine similarity

The system compares product text representations and recommends similar products.

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask
- Flask-SQLAlchemy

### Database
- SQLite

### AI / Machine Learning
- Pandas
- Scikit-learn TF-IDF Vectorizer
- NumPy
- Content-based recommendation

### Development Tools
- Git
- GitHub
- Python Virtual Environment

## Dataset

The project uses an e-commerce product dataset containing product information such as:

- Product ID
- Product name
- Product description
- Brand
- Variant
- Price
- Discounted price
- Usage
- Image URL

The active dataset is stored at:

`data/train.csv`

## Project Structure

```text
E-Commerce-Full-Stack-Platform-with-AI-Recommendations/
│
├── ai/
│   └── recommendation.py
│
├── data/
│   └── train.csv
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   ├── products.html
│   ├── product.html
│   ├── recommendations.html
│   ├── cart.html
│   ├── checkout.html
│   ├── login.html
│   ├── signup.html
│   └── order_success.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
