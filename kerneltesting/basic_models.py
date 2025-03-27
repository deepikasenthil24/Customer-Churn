from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np

#1
def random_forest_classifier(X, y):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X,y)
    
    prediction_rf = rf_clf.predict(X)
    return prediction_rf 


#2
def gradient_boosting_classifier(X, y):
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X, y)
    
    prediction_gb = gb_clf.predict(X)
    return prediction_gb

#3
def logistic_reg_model(X_train, y_train, X_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    return y_pred


#4 LSTM time series analysis
def LSTM_preds(X_train_processed, X_test_processed, y_train):
    def create_lstm_dataset(data, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps - 1):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps, 0])  # Assuming churn is the first column in y
        return np.array(X), np.array(y)

    # Preparing the data for LSTM (Training data)
    lstm_data = pd.DataFrame(X_train_processed)  # Convert training data to DataFrame
    lstm_data.columns = lstm_data.columns.astype(str)  # Ensure column names are strings
    lstm_data['Churn'] = y_train.values  # Add churn column to the DataFrame

    # Scale the features (excluding the churn column)
    scaler = StandardScaler()
    lstm_data_scaled = scaler.fit_transform(lstm_data.drop('Churn', axis=1))  # Drop 'Churn' before scaling

    # Set time_steps for LSTM
    time_steps = 10

    # Create LSTM dataset for training
    X_lstm, y_lstm = create_lstm_dataset(lstm_data_scaled, time_steps)

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, X_lstm.shape[2])))  # Adjust input shape
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))  # Use sigmoid activation for binary classification (churn prediction)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Reshape X_lstm for LSTM input
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
    
    # Train the model
    model.fit(X_lstm, y_lstm, epochs=20, batch_size=64)

    # Preparing the test data
    lstm_test_data = pd.DataFrame(X_test_processed)  # Convert test data to DataFrame
    lstm_test_data.columns = lstm_test_data.columns.astype(str)  # Ensure column names are strings

    # Scale the test data (excluding the churn column)
    lstm_test_data_scaled = scaler.transform(lstm_test_data)  # Use the same scaler fitted on training data
    
    # Create LSTM dataset for test data
    X_test_lstm, y_test_lstm = create_lstm_dataset(lstm_test_data_scaled, time_steps)
    
    # Reshape X_test_lstm for LSTM input
    X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], X_test_lstm.shape[2]))
    
    # Make predictions
    predictions = model.predict(X_test_lstm)
    
    # Convert predictions to binary outcomes
    threshold = 0.5 
    churn_predictions = (predictions > threshold).astype(int)
    
    return churn_predictions

