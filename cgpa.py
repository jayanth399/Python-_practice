import streamlit as st

st.title("🧮 Simple Calculator")

# Inputs
num1 = st.number_input("Enter first number:", value=0.0)
num2 = st.number_input("Enter second number:", value=0.0)
operation = st.selectbox("Choose operation", ["Addition", "Subtraction", "Multiplication", "Division"])

# Calculation
if st.button("Calculate"):
    if operation == "Addition":
        result = num1 + num2
    elif operation == "Subtraction":
        result = num1 - num2
    elif operation == "Multiplication":
        result = num1 * num2
    elif operation == "Division":
        result = "Error: Division by zero" if num2 == 0 else num1 / num2

    st.success(f"Result: {result}")
