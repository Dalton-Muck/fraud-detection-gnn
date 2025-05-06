# Filename: gnn_fraud_detection_complete.py
# Author: Dalton Muck
# Date: 2025-04-22
# Purpose: Comprehensive GNN for fraud detection with all edge cases

import pandas as pd
import random
import torch
import kagglehub
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import os
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import classification_report, confusion_matrix

def load_and_preprocess_data(path):
    """Load and preprocess the dataset with all original constraints"""
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the dataset directory.")
    
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    # Original preprocessing to match kaggles constraints
    df['fraud_bool'] = df['fraud_bool'].apply(lambda x: max(0, min(1, x)))
    df['income'] = df['income'].apply(lambda x: max(0, min(1, x)))
    df['name_email_similarity'] = df['name_email_similarity'].apply(lambda x: max(0, min(1, x)))
    df['prev_address_months_count'] = df['prev_address_months_count'].apply(lambda x: max(-1, min(380, x)))
    df['current_address_months_count'] = df['current_address_months_count'].apply(lambda x: max(-1, min(406, x)))
    df['customer_age'] = df['customer_age'].apply(lambda x: max(0, x))
    df['days_since_request'] = df['days_since_request'].apply(lambda x: max(0, min(78, x)))
    df['intended_balcon_amount'] = df['intended_balcon_amount'].apply(lambda x: max(-1, min(108, x)))
    df['zip_count_4w'] = df['zip_count_4w'].apply(lambda x: max(1, min(5767, x)))
    
    # Add your original derived features
    df['total_address_months'] = df['prev_address_months_count'] + df['current_address_months_count']
    df['address_stability'] = df['current_address_months_count'] / (df['prev_address_months_count'] + 1)
    df['income_balance_ratio'] = df['income'] / (df['intended_balcon_amount'] + 1e-6)
    
    return df

def create_all_edge_cases(df):
    """Create all 6 original edge cases as specified"""
    edges = []
    
    # 1. Frequent address changes (original add_address_change_edges)
    frequent_changers = df[df['total_address_months'] < 7].index
    edges.extend([(idx, 'frequent_address_change') for idx in frequent_changers])
    print(f"Frequent address changes edges added: {len(frequent_changers)}")
    
    # 2. Missing address data (original add_missing_address_edges)
    missing_address = df[(df['prev_address_months_count'] == -1) & 
                        (df['current_address_months_count'] == -1)].index
    edges.extend([(idx, 'missing_address') for idx in missing_address])
    print(f"Missing address edges added: {len(missing_address)}")
    
    # 3. Low income-to-balance ratio (original add_income_to_balance_edges)
    low_income_high_balance = df[(df['income'] < 0.25) & 
                                (df['intended_balcon_amount'] > 75)].index
    edges.extend([(idx, 'low_income_high_balance') for idx in low_income_high_balance])
    print(f"Low income-to-balance ratio edges added: {len(low_income_high_balance)}")
    
    # 4. Low email similarity (original add_email_similarity_edges)
    low_similarity = df[df['name_email_similarity'] < 0.01].index
    edges.extend([(idx, 'low_email_similarity') for idx in low_similarity])
    print(f"Low email similarity edges added: {len(low_similarity)}")
    
    # 5. High zipcode activity (original add_zipcode_activity_edges)
    recent_activity = df[df['days_since_request'] <= 7]
    top_zips = recent_activity['zip_count_4w'].value_counts().nlargest(3).index
    high_activity = df[df['zip_count_4w'].isin(top_zips)].index
    edges.extend([(idx, 'high_zip_activity') for idx in high_activity])
    print(f"High zipcode activity edges added: {len(high_activity)}")
    
    # 6. Age-address mismatch (original add_age_address_edges)
    age_mismatch = df[df['customer_age'] < ((df['current_address_months_count'] / 12 + 
                                             df['prev_address_months_count'] / 12) - 9)].index
    edges.extend([(idx, 'age_address_mismatch') for idx in age_mismatch])
    print(f"Age-address mismatch edges added: {len(age_mismatch)}")
    
    # Add connections between similar nodes (enhancement)
    for case in ['frequent_address_change', 'missing_address', 'low_income_high_balance',
                'low_email_similarity', 'high_zip_activity', 'age_address_mismatch']:
        case_nodes = [e[0] for e in edges if e[1] == case]
        edges.extend(combinations(case_nodes, 2))
    
    return edges

def build_pyg_data(df):
    """Build PyTorch Geometric Data object with all features and edges"""
    # Select and scale features
    feature_columns = [
        'income', 'name_email_similarity', 'prev_address_months_count',
        'current_address_months_count', 'customer_age', 'days_since_request',
        'intended_balcon_amount', 'zip_count_4w', 'total_address_months',
        'address_stability', 'income_balance_ratio'
    ]
    
    # more feature selection
    # Replace invalid values (NaN, infinity) with appropriate defaults
    df[feature_columns] = df[feature_columns].replace([float('inf'), -float('inf')], float('nan')).fillna(0)
    
    scaler = StandardScaler()
    node_features = scaler.fit_transform(df[feature_columns])
    
    # Create all edges
    edges = create_all_edge_cases(df)
    
    # Create node mapping including flag nodes
    all_nodes = list(df.index) + [
        'frequent_address_change',
        'missing_address',
        'low_income_high_balance',
        'low_email_similarity',
        'high_zip_activity',
        'age_address_mismatch'
    ]
    node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Convert edges to indices
    edge_index = [[node_mapping[src], node_mapping[dst]] for src, dst in edges]
    
    # Create feature matrix with dummy features for flag nodes
    flag_features = torch.zeros((6, node_features.shape[1]))
    x = torch.cat([
        torch.tensor(node_features, dtype=torch.float),
        flag_features
    ])
    
    # Create labels (1 for flag nodes)
    y = torch.cat([
        torch.tensor(df['fraud_bool'].values, dtype=torch.long),
        torch.ones(6, dtype=torch.long)
    ])
    
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        y=y
    )

def main():
    """Main execution function"""
    # Download and load dataset
    path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    df = load_and_preprocess_data(path)
    
    # Build graph data
    graph_data = build_pyg_data(df)
    
    # Print statistics
    print(f"Created graph with:")
    print(f"- Nodes: {graph_data.num_nodes} (including 6 flag nodes)")
    print(f"- Edges: {graph_data.num_edges}")
    print(f"- Features: {graph_data.num_features}")
    print(f"- Fraud rate: {df['fraud_bool'].mean():.2%}")
    
    # Train-test split
    train_mask = torch.rand(graph_data.num_nodes - 6) < 0.80  # 80% for training
    # ensures nodes in train set are not in test set
    test_mask = ~train_mask
    train_mask = torch.cat([train_mask, torch.zeros(6, dtype=torch.bool)])  # Exclude flag nodes
    test_mask = torch.cat([test_mask, torch.zeros(6, dtype=torch.bool)])    # Exclude flag nodes
    
    # Define the GNN model
    import torch.nn.functional as F
    class GNNModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    # Initialize model, optimizer, and loss function
    # 16 is the number of patterns to learn from the data, increasing this number increases the complexity of the model
    model = GNNModel(graph_data.num_features, 16, 2)  # 2 output classes: fraud or not
    fraud_weight = (1 / df['fraud_bool'].mean()) * .8  # Scale down slightly
    loss_fn = torch.nn.NLLLoss(weight=torch.tensor([1.0, fraud_weight], dtype=torch.float))

    # Oversample fraud cases in training
    fraud_indices = torch.where(graph_data.y[train_mask] == 1)[0]
    non_fraud_indices = torch.where(graph_data.y[train_mask] == 0)[0]
    # create more training data by repeating the fraud cases
    oversampled_indices = torch.cat([non_fraud_indices, fraud_indices.repeat(20)])  # 20x oversampling
    train_mask = torch.zeros_like(train_mask)
    train_mask[oversampled_indices] = True
    # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Training loop
    # epoche: number of iterations over the entire dataset updates the model weights after each epoch
    for epoch in range(100):  # Loop over 100 epochs
        model.train()
        # reset: need to call this before the backwards pass to reset gradients which control the weight updates
        optimizer.zero_grad()
        # forward pass: gets informationfrom neighboring nodes which is passed through the weight metrix
        out = model(graph_data.x, graph_data.edge_index) 
        # loss: measures how well the model peformed by comparing train and test data
        loss = loss_fn(out[train_mask], graph_data.y[train_mask])
        # adjusts weights in way that minimizes the loss function, uses gradients and chain rule
        loss.backward() 
        # update weights: updates the model parameters using the optimizer
        optimizer.step()
        
        if epoch % 10 == 0: 
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate the model and return predictions
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        # Get predicted probabilities for fraud class (class 1)
        fraud_probs = torch.exp(out[:, 1])  # Convert log-probs to probabilities
        
        # Create a DataFrame with results
        results = pd.DataFrame({
            'node_index': range(len(graph_data.y[:len(df)])),  # Exclude flag nodes
            'actual_fraud': graph_data.y[:len(df)].numpy(),  # Actual labels
            'predicted_fraud_prob': fraud_probs[:len(df)].numpy(),  # Fraud probability
        })
        
        # Add the original data columns for context
        results = pd.concat([results, df.reset_index(drop=True)], axis=1)
        
        # Calculate evaluation metrics
        pred_labels = (fraud_probs[:len(df)] > 0.8).numpy().astype(int)  # Use threshold of 0.5 for classification
        print("\nClassification Report:")
        print(classification_report(results['actual_fraud'], pred_labels))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(results['actual_fraud'], pred_labels))
        
        return results

if __name__ == "__main__":
    predictions = main()
    print("\nTop 10 predicted fraud probabilities:")
    print(predictions[['node_index', 'actual_fraud', 'predicted_fraud_prob']].nlargest(10, 'predicted_fraud_prob'))
    
    # Save predictions to CSV
    predictions.to_csv('fraud_predictions.csv', index=False)
    print("\nPredictions saved to fraud_predictions.csv")
