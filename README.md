import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Petrol Pump Billing System",
    page_icon="⛽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Fuel Prices (can be updated here) ---
FUELS = {
    "Petrol": 110.50,
    "Diesel": 95.75,
    "Power Petrol": 115.25
}


# --- Functions ---

def calculate_bill(fuel_type, litres):
    """
    Computes subtotal, discount, and final amount based on fuel type and quantity.

    Args:
        fuel_type (str): The type of fuel selected.
        litres (float): The quantity of fuel in litres.

    Returns:
        tuple: A tuple containing subtotal, discount, and final_amount.
    """
    if fuel_type not in FUELS:
        return None, None, None

    price_per_litre = FUELS[fuel_type]
    subtotal = litres * price_per_litre

    # Apply discount for bulk purchase
    discount = 0
    if litres > 20:
        discount = subtotal * 0.05  # 5% discount for more than 20 litres

    final_amount = subtotal - discount

    return subtotal, discount, final_amount


# --- UI Layout ---

# --- Sidebar for Inputs ---
with st.sidebar:
    st.image("https://placehold.co/400x200/2B3467/FFFFFF?text=Fuel+Station&font=inter", use_column_width=True)
    st.header("Fuel Options")

    # Display Fuel Prices Dynamically
    st.markdown("##### Current Fuel Prices:")
    for fuel, price in FUELS.items():
        st.markdown(f"- **{fuel}:** ₹ {price:.2f}/litre")
    st.markdown("---")

    # Input Fields
    st.header("Enter Details")
    fuel_type_selection = st.radio(
        "Select Fuel Type:",
        options=list(FUELS.keys()),
        key="fuel_type"
    )

    litres_input = st.number_input(
        "Enter Quantity (in litres):",
        min_value=0.1,
        max_value=200.0,
        value=1.0,
        step=0.1,
        help="Select the amount of fuel you want to purchase (0.1 to 200 litres)."
    )

    generate_bill_button = st.button("Generate Bill", use_container_width=True, type="primary")

# --- Main Content Area for Displaying Receipt ---
st.title("⛽ Petrol Pump Billing System")
st.markdown("Welcome! Please enter your fuel details in the sidebar to generate a receipt.")

if generate_bill_button:
    if litres_input > 0:
        subtotal, discount, final_amount = calculate_bill(fuel_type_selection, litres_input)

        with st.spinner('Generating your receipt...'):
            time.sleep(1)

        st.balloons()
        st.header("Transaction Receipt")

        # Using columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Fuel Type", value=fuel_type_selection)
            st.metric(label="Quantity", value=f"{litres_input:.2f} litres")

        with col2:
            st.metric(label="Rate", value=f"₹ {FUELS[fuel_type_selection]:.2f} /litre")
            st.metric(label="Subtotal", value=f"₹ {subtotal:.2f}")

        st.markdown("---")

        if discount > 0:
            st.success(f"🎉 You received a discount of **₹{discount:.2f}** for purchasing over 20 litres!")

        st.subheader(f"Total Amount Payable: ₹ {final_amount:.2f}")

        st.info("Thank you for your purchase! Drive safe! 🚗")

    else:
        st.error("Please enter a valid quantity of litres.")

st.markdown("""
<style>
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 0.5rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
