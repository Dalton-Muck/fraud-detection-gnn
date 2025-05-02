# Filename : gnn.py
# Author :  Dalton Muck
# Date :    2025-04-22
# Purpose : Graph Neural Network (GNN) for fraud detection
import kagglehub
import pandas as pd
import os
import networkx as nx
def load_and_display_dataset(path):
    """
    Load the dataset from the given path and display its contents.
    Args:
        path (str): Path to the dataset directory.
    Returns:
        pd.DataFrame or None: The loaded DataFrame if a CSV file is found, otherwise None.
    """
    print("Path to dataset files:", path)
    # Assuming the dataset contains a CSV file, find the first CSV file in the directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        dataset_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(dataset_path)
        print(df.head())
        return df
    else:
        print("No CSV files found in the dataset directory.")
        return None
# Download latest version
# https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 
path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
df = load_and_display_dataset(path)

# Flags to add edge for 
def add_address_change_edges(df, graph):
    """
    Add edges based on frequent address changes.
    """
    count = 0
    for idx, row in df.iterrows():
        if row['prev_address_months_count'] + row['current_address_months_count'] < 7:
            graph.add_edge(f"node_{idx}", f"address_change_flag")
            count += 1
    print(f"Added {count} edges for frequent address changes.")
    return count
def add_missing_address_edges(df, graph):
    """
    Add edges for missing address data.
    """
    count = 0
    for idx, row in df.iterrows():
        if row['prev_address_months_count'] == -1 and row['current_address_months_count'] == -1:
            graph.add_edge(f"node_{idx}", f"missing_address_flag")
            count += 1
    print(f"Added {count} edges for missing address data.")
    return count
def add_income_to_balance_edges(df, graph):
    """
    Add edges based on low income-to-balance ratio.
    """
    count = 0
    for idx, row in df.iterrows():
        if row['income'] < 0.25 and row['intended_balcon_amount'] > 75:
            graph.add_edge(f"node_{idx}", f"low_income_high_balance_flag")
            count += 1
    print(f"Added {count} edges for low income-to-balance ratio.")
    return count

def add_email_similarity_edges(df, graph):
    """
    Add edges for low email similarity.
    """
    count = 0
    for idx, row in df.iterrows():
        if row['name_email_similarity'] < 0.01:
            graph.add_edge(f"node_{idx}", f"low_email_similarity_flag")
            count += 1
    print(f"Added {count} edges for low email similarity.")
    return count
def add_zipcode_activity_edges(df, graph):
    """
    Add edges for high activity in a new zipcode.
    """
    count = 0
    recent_activity_counter = {}
    # Count the total number of accounts added in the last 7 days for each zipcode
    for idx, row in df.iterrows():
        zipcode = row['zip_count_4w']
        if zipcode not in recent_activity_counter:
            recent_activity_counter[zipcode] = {'count': 0, 'total': 0}
        recent_activity_counter[zipcode]['total'] += 1
        if row['days_since_request'] <= 7:
            recent_activity_counter[zipcode]['count'] += 1

    # Sort zipcodes by activity ratio (count / total) in descending order
    sorted_zipcodes = sorted(
        recent_activity_counter.items(),
        key=lambda x: x[1]['count'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )

    # Get the top 3 zipcodes with the highest activity ratio
    top_zipcodes = {zipcode for zipcode, _ in sorted_zipcodes[:3]}
    print(f"Top 3 zipcodes with the most activity: {top_zipcodes}")

    # Add edges to the graph for nodes in the top 3 zipcodes
    for idx, row in df.iterrows():
        if row['zip_count_4w'] in top_zipcodes and row['days_since_request'] < 7:
            graph.add_edge(f"node_{idx}", f"high_activity_zipcode_flag")
            count += 1

    print(f"Added {count} edges for high activity in a new zipcode.")
    return count

def add_age_address_edges(df, graph):
    """
    Add edges for age mismatch with address age.
    """
    count = 0
    for idx, row in df.iterrows():
        if row['customer_age'] < (row['current_address_months_count'] / 12 + row['prev_address_months_count'] / 12) - 9:
            graph.add_edge(f"node_{idx}", f"age_address_mismatch_flag")
            count += 1
    print(f"Added {count} edges for age and address mismatch.")
    return count
def main(df):
    """
    Main function to add edges to the graph based on various conditions.
    """
    graph = nx.Graph()
    total_edges = 0
    total_edges += add_address_change_edges(df, graph)
    total_edges += add_missing_address_edges(df, graph)
    total_edges += add_income_to_balance_edges(df, graph)
    total_edges += add_email_similarity_edges(df, graph)
    total_edges += add_zipcode_activity_edges(df, graph)
    total_edges += add_age_address_edges(df, graph)
    #print the total number of rows with fraud_bool = 1
    fraud_count = df[df['fraud_bool'] == 1].shape[0]
    print(f"Total number of fraud cases: {fraud_count}")
    print(f"Total edges added: {total_edges}")
    return graph
# Create the graph using the dataframe
graph = main(df)