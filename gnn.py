# Filename: gnn_fraud_detection_fast.py
import pandas as pd
import torch
import kagglehub
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_and_sample_data(path, sample_size=20000, fraud_oversample=5):
    """Load and sample the dataset"""
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    # Sample data while preserving fraud cases
    fraud_df = df[df['fraud_bool'] == 1]
    non_fraud_df = df[df['fraud_bool'] == 0].sample(n=min(sample_size, len(df)), random_state=42)
    
    # Oversample fraud cases
    fraud_sample = fraud_df.sample(n=min(len(fraud_df), sample_size//fraud_oversample), 
                            replace=True, random_state=42)
    
    return pd.concat([non_fraud_df, fraud_sample]).sample(frac=1, random_state=42)

def build_pyg_data(df, k_neighbors=20):
    """Build graph data using KNN for efficient edge creation"""
    # Feature engineering
    df['risk_score'] = (
        0.4 * (1 - df['name_email_similarity']) + 
        0.3 * (df['zip_count_4w'] / (df['days_since_request'] + 1)) + 
        0.3 * (df['intended_balcon_amount'] / (df['income'] + 1e-6)))
    
    # Add features that distinguish false positives
    median_zip_count = df['zip_count_4w'].median()
    df['high_risk_combo'] = ((df['name_email_similarity'] < 0.1) & 
                             (df['zip_count_4w'] > median_zip_count))
    
    feature_columns = [
        'income', 'name_email_similarity', 'prev_address_months_count',
        'current_address_months_count', 'customer_age', 'risk_score',
        'high_risk_combo'
    ]
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_columns].fillna(0))
    x = torch.tensor(features, dtype=torch.float)
    
    # Create edges using KNN
    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
    knn.fit(features)
    _, indices = knn.kneighbors(features)
    
    # Create edge_index
    edge_index = []
    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:  # No self-loops
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph
    
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        y=torch.tensor(df['fraud_bool'].values, dtype=torch.long)
    )

class FastFraudGNN(torch.nn.Module):
    """GNN model for fraud detection without second attention layer"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        
        # Second GCN layer
        x = F.relu(self.conv2(x, edge_index))
        
        # Fully connected layer for output
        x = F.dropout(x, p=0.3, training=self.training)
        return F.log_softmax(self.fc(x), dim=1)

def main():
    # Load and sample data
    path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    df = load_and_sample_data(path, sample_size=20000)
    graph_data = build_pyg_data(df)
    
    print(f"Graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    print(f"Fraud rate in sample: {df['fraud_bool'].mean():.2%}")
    
    # Train-test split
    indices = torch.randperm(len(df))
    train_size = int(0.8 * len(df))
    train_mask = indices[:train_size]
    test_mask = indices[train_size:]
    
    # Model setup
    model = FastFraudGNN(
        in_channels=graph_data.x.size(1),
        hidden_channels=32,
        out_channels=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fraud_weight = (1 / df['fraud_bool'].mean()) * 0.80
    weight = torch.tensor([1.0, fraud_weight], device=graph_data.x.device, dtype=torch.float)
    criterion = torch.nn.NLLLoss(weight=weight)
    
    # Training loop
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        probs = torch.exp(logits[:, 1]).numpy()
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(
            graph_data.y[test_mask].numpy(), 
            probs[test_mask.numpy()]
        )
        optimal_idx = np.argmax((2 * precision * recall) / (precision + recall + 1e-8))
        optimal_threshold = thresholds[optimal_idx]
        new_threshold = optimal_threshold * 1.3  # Adjust threshold to reduce false positives
        
        
        # Results
        preds = (probs >= new_threshold).astype(int)
        print("\nClassification Report:")
        print(classification_report(graph_data.y[test_mask].numpy(), preds[test_mask.numpy()]))
        
        # Save predictions
        results = pd.DataFrame({
            'actual': graph_data.y.numpy(),
            'predicted': preds,
            'probability': probs
        })
        results = pd.concat([results, df.reset_index(drop=True)], axis=1)
        results.to_csv('fraud_predictions_fast.csv', index=False)
        
        print("\nTop 10 predicted fraud cases:")
        print(results.nlargest(10, 'probability')[['actual', 'probability']])

if __name__ == "__main__":
    main()