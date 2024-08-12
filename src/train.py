import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

def main(train_file_path):
    # Load the preprocessed data from the Parquet file
    data = pd.read_parquet(train_file_path)

    # Prepare the features (X) and the target variable (y)
    X = data[['store_id', 'year', 'month', 'day', 'weekday']]
    y = data['total_sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=195)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=195)

    # Train the model on the training data
    print("Training model!")
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model evaluation complete. Mean Squared Error: {mse:.2f}")

    # Save the trained model to a pickle file
    model_save_path = '../models/model-2023-08-01.pickle'
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_save_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <path_to_preprocessed_data>")
    else:
        train_file_path = sys.argv[1]
        main(train_file_path)