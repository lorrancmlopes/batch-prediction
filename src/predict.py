import sys
import pandas as pd
import pickle

def main(model_path, data_path):
    # Load the trained model from the pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the data for prediction from the Parquet file
    data = pd.read_parquet(data_path)

    # Prepare the features (X) for prediction
    X = data[['store_id', 'year', 'month', 'day', 'weekday']]

    # Make predictions using the loaded model
    print("Making predictions!")
    predictions = model.predict(X)

    # Add the predictions as a new column to the DataFrame
    data['predicted_sales'] = predictions

    # Save the predictions to a new Parquet file
    output_path = '../data/predict-done-2023-08-03.parquet'
    data.to_parquet(output_path, index=False)

    print(f"Predictions saved to {output_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 predict.py <path_to_model> <path_to_data_for_prediction>")
    else:
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        main(model_path, data_path)
