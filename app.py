import streamlit as st
import joblib
import pandas as pd

# 1. Load the Model and Features
# Make sure these files are in your "Customer churn" folder
try:
    model = joblib.load('churn_model.pkl')
    features = joblib.load('feature_columns.pkl')
except:
    st.error("Model files not found. Run your training code first!")

# 2. Page Configuration
st.set_page_config(page_title="Churn & Retention Dashboard", layout="wide")
st.title("📊 Customer Churn & Retention Strategy Recommender")
st.markdown("Use this tool to identify high-risk customers and get instant retention action plans.")

# 3. Sidebar Input
st.sidebar.header("Step 1: Input Customer Data")
freq = st.sidebar.number_input("Frequency (Total Orders)", min_value=1, value=5)
monetary = st.sidebar.number_input("Monetary (Total Spend $)", min_value=1.0, value=250.0)
stock = st.sidebar.number_input("Product Diversity (Unique Items)", min_value=1, value=12)

# Calculation for the model
aov = monetary / freq

# 4. Prediction Logic
if st.button("Analyze Customer Risk"):
    # Prepare data
    input_df = pd.DataFrame([[freq, monetary, stock, aov]], columns=features)
    
    # Run Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Display Results
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"### RISK STATUS: HIGH RISK")
            st.write(f"**Churn Probability:** {prob*100:.1f}%")
        else:
            st.success(f"### RISK STATUS: LOW RISK")
            st.write(f"**Churn Probability:** {prob*100:.1f}%")

    # 5. The "Free" Retention Strategy Engine
    with col2:
        st.subheader("💡 Actionable Strategy")
        
        if prediction == 1:
            if monetary > 1000:
                st.warning("**Strategy: VIP Recovery**")
                st.write("This is a high-value customer. **Action:** Assign a dedicated account manager to call them personally with a 25% 'Premium Loyalty' credit.")
            elif freq > 20:
                st.warning("**Strategy: Frequency Boost**")
                st.write("This was a frequent shopper who stopped. **Action:** Send a 'We Miss You' automated email featuring their most-purchased product categories.")
            else:
                st.info("**Strategy: Standard Re-engagement**")
                st.write("General churn risk. **Action:** Include in the next bulk discount email blast (10% off coupon).")
        else:
            st.info("**Strategy: Relationship Maintenance**")
            st.write("Customer is healthy. **Action:** No aggressive discount needed. Send regular 'New Arrival' updates to keep them engaged.")

st.divider()
st.caption("Built with Python, XGBoost, and Streamlit | Dataset: UCI Online Retail")