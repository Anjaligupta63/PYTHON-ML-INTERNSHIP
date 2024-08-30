import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Define a fixed path for the CSV file
FILE_PATH = 'breast-cancer-data.csv'

# Streamlit app
def main():
    st.title('Breast Cancer Recurrence Prediction')

    # Step 1: Load the dataset from CSV
    data = pd.read_csv(FILE_PATH)

    # Display the dataset in the Streamlit app for reference
    st.write("Dataset Preview:")
    st.write(data.head())

    # Step 2: Data Preprocessing
    # Remove unwanted quotes from the data
    for column in data.select_dtypes(include=[object]).columns:
        data[column] = data[column].str.strip("'")

    # Define the features and the target variable
    X = data.drop('class', axis=1)
    y = data['class'].map(lambda x: 1 if x == 'recurrence-events' else 0)  # Adjust mapping if necessary
   # Step 3: One-Hot Encoding for categorical variables
    categorical_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Step 4: Create a pipeline that preprocesses the data, scales it, and applies logistic regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),  # with_mean=False due to one-hot encoding
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Train the model
    pipeline.fit(X_train, y_train)

    # Step 7: Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User input fields
    age = st.selectbox('Age', sorted(X['age'].unique()))
    menopause = st.selectbox('Menopause', sorted(X['menopause'].unique()))
    tumor_size = st.selectbox('Tumor Size', sorted(X['tumor-size'].unique()))
    inv_nodes = st.selectbox('Involved Nodes', sorted(X['inv-nodes'].unique()))
    node_caps = st.selectbox('Node Caps', sorted(X['node-caps'].unique()))
    deg_malig = st.selectbox('Degree of Malignancy', sorted(X['deg-malig'].unique()))
    breast = st.selectbox('Breast', sorted(X['breast'].unique()))
    breast_quad = st.selectbox('Breast Quadrant', sorted(X['breast-quad'].unique()))
    irradiate = st.selectbox('Irradiate', sorted(X['irradiate'].unique()))

    # Button to make a prediction
    if st.button('Predict'):
        # Create a DataFrame for the user input
        input_data = pd.DataFrame({
            'age': [age],
            'menopause': [menopause],
            'tumor-size': [tumor_size],
            'inv-nodes': [inv_nodes],
            'node-caps': [node_caps],
            'deg-malig': [deg_malig],
            'breast': [breast],
            'breast-quad': [breast_quad],
            'irradiate': [irradiate]
        })
    # Preprocess the input data
        input_data_processed = pipeline.named_steps['preprocessor'].transform(input_data)
        input_data_scaled = pipeline.named_steps['scaler'].transform(input_data_processed)

        # Make a prediction using the trained model
        prediction = pipeline.named_steps['classifier'].predict(input_data_scaled)

        # Display the prediction result
        if prediction[0] == 1:
            st.markdown('### Predicted: Recurrence-Events (Cancerous)')
        else:
            st.markdown('### Predicted: No-Recurrence-Events (Non-Cancerous)')

if _name_ == "_main_":
    main()
