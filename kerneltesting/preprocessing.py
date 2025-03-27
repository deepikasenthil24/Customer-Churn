import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def preprocess_data(data):
    df = pd.read_csv(data, encoding='ISO-8859-1')
    
    # Convert date columns to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    
    # Calculate the days since the first order for each customer
    df['Days Since First Order'] = df.groupby('Customer ID')['Order Date'].transform(lambda x: (x - x.min()).dt.days)

    # Create lag features for 'Sales' and 'Profit'
    df['lag_sales_1'] = df.groupby('Customer ID')['Sales'].shift(1)
    df['lag_profit_1'] = df.groupby('Customer ID')['Profit'].shift(1)
    
    # Create rolling statistics for 'Sales' and 'Profit'
    df['rolling_sales_mean_3'] = df.groupby('Customer ID')['Sales'].rolling(window=3).mean().reset_index(0, drop=True)
    df['rolling_profit_mean_3'] = df.groupby('Customer ID')['Profit'].rolling(window=3).mean().reset_index(0, drop=True)

    # Create cohort feature: Months since the first order for each customer
    #data['cohort'] = (data['Order Date'] - data.groupby('Customer ID')['Order Date'].transform('min')).dt.days
    df['cohort'] = (df['Order Date'].dt.to_period('M') - df.groupby('Customer ID')['Order Date'].transform('min').dt.to_period('M')).apply(lambda x: x.n)


    # Create target label 'churn': Mark customers who haven't made an order in the last 90 days as churned
    latest_order_date = df['Order Date'].max()
    df['churn'] = np.where((latest_order_date - df.groupby('Customer ID')['Order Date'].transform('max')).dt.days > 90, 1, 0)

    
    data_processed_prenan = df#preprocess_data(data)
    
    data_processed = data_processed_prenan.dropna() # Drop rows with missing values caused by shifting or rolling
    
    data_processed['Order Year'] = data_processed['Order Date'].dt.year
    data_processed['Order Month'] = data_processed['Order Date'].dt.month
    data_processed['Ship Year'] = data_processed['Ship Date'].dt.year
    data_processed['Ship Month'] = data_processed['Ship Date'].dt.month
    data_processed['Shipping Delay'] = (data_processed['Ship Date'] - data_processed['Order Date']).dt.days
    
    
    categorical_columns = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category', 'Postal Code', 'Product Name']
    numeric_columns = ['Sales', 'Quantity', 'Discount', 'Profit', 'lag_sales_1', 'lag_profit_1', 'rolling_sales_mean_3', 'rolling_profit_mean_3', 'Days Since First Order', 'cohort',  'Order Year', 'Order Month', 'Ship Year', 'Ship Month', 'Shipping Delay',  'Days Since First Order']
    
    
    # One-Hot Encoding for categorical features with unknown category handling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])

    X = data_processed.drop(['churn', 'Row ID', 'Order ID', 'Customer Name', 'Product ID', 'Order Date', 'Ship Date',], axis=1)
    y = data_processed['churn']
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Apply the preprocessing pipeline to the training and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    
    return data_processed, X_train_processed, X_test_processed, y_train, y_test
        

