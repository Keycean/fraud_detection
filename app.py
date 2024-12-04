import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Initialize session state for storing transactions 
if 'transactions' not in st.session_state:
    # Create initial sample dataset
    st.session_state.transactions = pd.DataFrame({
        'amount': [150.25, 75.80, 2500.00, 45.99, 89.99, 1850.75, 125.50, 95.25, 3500.00, 
                  65.99, 85.50, 2750.25, 115.75, 195.99, 4500.00],
        'time': [8.5, 9.25, 2.75, 12.5, 13.33, 3.25, 15.75, 16.5, 2.33, 
                 18.25, 19.75, 1.5, 21.33, 22.25, 2.75],
        'location': [0.2, 0.3, 1.8, 0.4, 0.1, 1.7, 0.5, 0.2, 1.9, 
                    0.3, 0.4, 1.8, 0.2, 0.5, 1.9]
    })

def detect_anomalies(data, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data[['amount', 'time', 'location']])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features_scaled)
    return clusters

def main():
    st.title("Credit Card Fraud Detection with DBSCAN")
    st.write("Analyze credit card transactions for potential fraud using DBSCAN clustering.")
    
    # Sidebar controls for DBSCAN parameters
    st.sidebar.header("Model Parameters")
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)
    
    # Add new transaction section
    st.sidebar.header("Add New Transaction")
    amount = st.sidebar.number_input("Amount ($)", min_value=0.0, max_value=10000.0, value=100.0)
    time = st.sidebar.number_input("Time (24-hour format)", min_value=0.0, max_value=24.0, value=12.0)
    location = st.sidebar.number_input("Location (-1 to 2)", min_value=-1.0, max_value=2.0, value=0.0)
    
    if st.sidebar.button("Add Transaction"):
        new_transaction = pd.DataFrame({
            'amount': [amount],
            'time': [time],
            'location': [location]
        })
        st.session_state.transactions = pd.concat([st.session_state.transactions, new_transaction], ignore_index=True)
        st.sidebar.success("Transaction added!")

    # Clear data button
    if st.sidebar.button("Clear All Transactions"):
        st.session_state.transactions = pd.DataFrame(columns=['amount', 'time', 'location'])
        st.sidebar.success("All transactions cleared!")
    
    # Display current transactions and analysis
    if not st.session_state.transactions.empty:
        # Detect anomalies
        clusters = detect_anomalies(st.session_state.transactions, eps, min_samples)
        data = st.session_state.transactions.copy()
        data['status'] = np.where(clusters == -1, 'Potential Fraud', 'Normal')
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Statistics")
            total_transactions = len(data)
            fraud_transactions = (clusters == -1).sum()
            
            st.metric("Total Transactions", total_transactions)
            st.metric("Potential Fraud Detected", fraud_transactions)
            st.metric("Fraud Percentage", f"{(fraud_transactions/total_transactions)*100:.2f}%")
            
            # Show all transactions
            st.subheader("All Transactions")
            st.dataframe(data)
        
        with col2:
            st.subheader("Visualization")
            # 3D scatter plot
            fig = px.scatter_3d(data, 
                               x='amount', 
                               y='time', 
                               z='location',
                               color='status',
                               title="Transaction Clustering Results",
                               color_discrete_map={'Normal': 'blue', 'Potential Fraud': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction amount distribution
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(data, 
                          x='amount', 
                          color='status',
                          barmode='overlay',
                          title="Distribution of Transaction Amounts",
                          color_discrete_map={'Normal': 'blue', 'Potential Fraud': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No transactions available. Please add transactions using the sidebar.")

if __name__ == "__main__":
    main()