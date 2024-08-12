import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('../data/train-2023-08-01.csv')

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Group by 'store_id' and 'date' to calculate the total sales per store per day
grouped_df = df.groupby(['store_id', 'date']).agg(total_sales=('price', 'sum')).reset_index()

# Extract year, month, day, and weekday from the 'date' column
grouped_df['year'] = grouped_df['date'].dt.year
grouped_df['month'] = grouped_df['date'].dt.month
grouped_df['day'] = grouped_df['date'].dt.day
grouped_df['weekday'] = grouped_df['date'].dt.weekday

# Drop the 'date' column as it's no longer needed
grouped_df = grouped_df.drop(columns=['date'])

# Save the result to a Parquet file in the '../data' folder
grouped_df.to_parquet('../data/processed_sales_data.parquet', index=False)

print("Processing complete. Data saved to '../data/processed_sales_data.parquet'.")