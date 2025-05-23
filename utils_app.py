import streamlit as st
import os
import json

import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn

from dataset.loaders import DataProvider, get_dataset_info, get_available_datasets


# Define constants
DATASET_DIR = './dataset/'
RESULTS_DIR = './results/'
LOGS_DIR = './logs/'
CHECKPOINTS_DIR = './checkpoints/'
MODELS_DIR = './models/'

# Ensure directories exist
for directory in [DATASET_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_dataset_list():
    """Get list of available datasets"""
    return get_available_datasets()

def get_dataset_info(dataset_id):
    """Get information about a specific dataset"""
    return get_dataset_info(dataset_id)

def load_dataset(dataset_id):
    """Load a dataset using the DataProvider"""
    args = get_args()
    args.root_path = DATASET_DIR
    args.data = dataset_id
    args.data_path = f'{DATASET_DIR}/PdM_{dataset_id}'
    args.features = None
    args.target = None
    args.batch_size = 32
    args.num_workers = 0
    args.normalize = True
    
    provider = DataProvider(args)
    return provider

def plot_dataset(dataset_id):
    """Plot dataset visualizations"""
    provider = load_dataset(dataset_id)
    train_dataset, _ = provider.get_data('TRAIN')
    
    if train_dataset is None:
        st.error("Failed to load dataset")
        return
    
    # Time series plot
    st.subheader("Time Series Plot")
    n_features = train_dataset.data.shape[-1]
    feature_idx = st.selectbox(
        "Select Feature",
        range(n_features),
        format_func=lambda x: f"Feature {x+1}"
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=train_dataset.data[0, :, feature_idx],
        mode='lines',
        name='Sample 1'
    ))
    fig.update_layout(
        title=f"Time Series Plot - Feature {feature_idx+1}",
        xaxis_title="Time Step",
        yaxis_title="Value"
    )
    st.plotly_chart(fig)
    
    # Feature distribution
    st.subheader("Feature Distribution")
    fig = px.histogram(
        x=train_dataset.data[:, :, feature_idx].flatten(),
        title=f"Distribution - Feature {feature_idx+1}"
    )
    st.plotly_chart(fig)
    
    # Label distribution if available
    if train_dataset.labels is not None:
        st.subheader("Label Distribution")
        fig = go.Figure(data=[
            go.Bar(
                x=train_dataset.class_names,
                y=[(train_dataset.labels == i).sum().item() for i in range(len(train_dataset.class_names))],
                text=[(train_dataset.labels == i).sum().item() for i in range(len(train_dataset.class_names))],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Label Distribution",
            xaxis_title="Class",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)

def get_model_list():
    """Get list of available models"""
    return ["LSTM", "GRU", "Transformer"]

def get_config_options():
    """Get model configuration options"""
    return {
        "batch_size": [16, 32, 64, 128],
        "learning_rate": [0.0001, 0.001, 0.01],
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2, 3, 4]
    }

def run_model(dataset_id, model_type, config):
    """Run model training"""
    provider = load_dataset(dataset_id)
    train_dataset, train_loader = provider.get_data('TRAIN')
    val_dataset, val_loader = provider.get_data('VAL')
    test_dataset, test_loader = provider.get_data('TEST')
    
    if train_loader is None:
        st.error("Failed to load dataset")
        return None
    
    # Initialize model
    input_size = train_dataset.data.shape[-1]
    model = create_model(
        model_type,
        input_size,
        config["hidden_size"],
        config["num_layers"]
    )
    
    # Training loop
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update progress
        progress = (epoch + 1) / config["epochs"]
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Test evaluation
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
        "accuracy": accuracy
    }

def visualize_training(results):
    """Visualize training results"""
    if results is None:
        return
    
    # Plot training curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=results["train_losses"],
        mode='lines',
        name='Training Loss'
    ))
    fig.add_trace(go.Scatter(
        y=results["val_losses"],
        mode='lines',
        name='Validation Loss'
    ))
    fig.update_layout(
        title="Training Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )
    st.plotly_chart(fig)
    
    # Display test results
    st.subheader("Test Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Loss", f"{results['test_loss']:.4f}")
    with col2:
        st.metric("Accuracy", f"{results['accuracy']:.2f}%")

def load_results(dataset_id, model_type):
    """Load saved results"""
    results_path = f"{RESULTS_DIR}/{dataset_id}/{model_type}/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def create_model(model_type, input_size, hidden_size, num_layers):
    """Create a model based on the selected type"""
    if model_type == "LSTM":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    elif model_type == "GRU":
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    elif model_type == "Transformer":
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=4,
                dim_feedforward=hidden_size
            ),
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")