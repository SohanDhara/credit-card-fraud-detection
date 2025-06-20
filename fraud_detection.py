# This code is a Streamlit app for fraud detection using a pre-trained model.
import streamlit as st
import pandas as pd
import joblib
import sklearn

# Try loading the model 
model = joblib.load('fraud_detection_pipeline.pkl')

# Check if the model is loaded successfully
st.title("Fraud Detection App")
st.markdown("Please enter the transaction details and use the predict button")
st.divider()

# Input fields for transaction details
transcation_type = st.selectbox("Transaction Type", ["CASH WITHDRAW", "TRANSFER", "DEPOSIT", "PAYMENT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transcation_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    # Ensure the model is loaded and ready for prediction
    prediction = model.predict(input_data)[0]
    
    # Display the prediction result
    st.subheader(f"Prediction Result: '{int(prediction)}'")
    
    # Interpret the prediction
    if prediction == 1:
        st.error("The transaction is fraud.")
    else:
        st.success("The transaction is Legit.")
