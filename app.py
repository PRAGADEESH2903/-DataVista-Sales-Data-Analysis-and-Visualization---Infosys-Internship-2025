
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("combined_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("📦 Delivery Prediction App")
st.markdown("Predict whether the order will be delayed and if a refund will be requested.")

# User inputs
platform = st.selectbox("Platform", ["Amazon", "Flipkart", "BigBasket", "JioMart", "Other"])
delivery_time = st.number_input("Delivery Time (Minutes)", min_value=0)
product_category = st.selectbox("Product Category",
                                ["Fruits & Vegetables", "Electronics", "Groceries", "Clothing", "Other"])
order_value = st.number_input("Order Value (INR)", min_value=0)
feedback = st.selectbox("Customer Feedback", ["Excellent", "Good", "Average", "Poor"])
rating = st.slider("Service Rating (1 to 5)", 1, 5)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Platform": platform,
        "Delivery Time (Minutes)": delivery_time,
        "Product Category": product_category,
        "Order Value (INR)": order_value,
        "Customer Feedback": feedback,
        "Service Rating": rating
    }])

    prediction = model.predict(input_data)[0]
    delay, refund = prediction.split("_")

    st.success(f"📊 Prediction: Delivery = {delay}, Refund = {refund}")
