import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Define constants
LEADERBOARD_DIR = './leaderboard/'
os.makedirs(LEADERBOARD_DIR, exist_ok=True)

def load_leaderboard_data():
    """Load leaderboard data from JSON file"""
    leaderboard_file = os.path.join(LEADERBOARD_DIR, 'leaderboard.json')
    
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_leaderboard_data(data):
    """Save leaderboard data to JSON file"""
    leaderboard_file = os.path.join(LEADERBOARD_DIR, 'leaderboard.json')
    
    with open(leaderboard_file, 'w') as f:
        json.dump(data, f, indent=2)

def update_leaderboard(dataset_id, model, results):
    """Update leaderboard with new results"""
    # Load existing leaderboard data
    leaderboard_data = load_leaderboard_data()
    
    # Initialize dataset entry if not exists
    if dataset_id not in leaderboard_data:
        leaderboard_data[dataset_id] = {}
    
    # Create entry for model
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    leaderboard_data[dataset_id][model] = {
        "timestamp": timestamp,
        "results": results
    }
    
    # Save updated leaderboard
    save_leaderboard_data(leaderboard_data)
    
    # Update session state
    st.session_state.leaderboard_data = leaderboard_data

def display_leaderboard():
    """Display leaderboard data"""
    # Load leaderboard data
    if 'leaderboard_data' not in st.session_state:
        st.session_state.leaderboard_data = load_leaderboard_data()
    
    # Filters
    st.subheader("Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get list of all datasets in leaderboard
        dataset_options = list(st.session_state.leaderboard_data.keys())
        if not dataset_options:
            st.warning("No data available in leaderboard yet")
            return
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            dataset_options,
            index=0 if dataset_options else None
        )
    
    with col2:
        # Get metric options
        metric_options = ["accuracy", "loss", "f1"]
        selected_metric = st.selectbox(
            "Select Metric",
            metric_options,
            index=0
        )
    
    with col3:
        # Sort order
        sort_order = st.selectbox(
            "Sort Order",
            ["Best to Worst", "Worst to Best"],
            index=0
        )
        # For loss, we need to invert the sort order (lower is better)
        if selected_metric == "loss":
            is_ascending = sort_order == "Worst to Best"
        else:
            is_ascending = sort_order != "Worst to Best"
    
    # Prepare data for display
    if selected_dataset in st.session_state.leaderboard_data:
        dataset_results = st.session_state.leaderboard_data[selected_dataset]
        
        # Convert to DataFrame for easier processing
        data = []
        for model, entry in dataset_results.items():
            if "results" in entry and selected_metric in entry["results"]:
                data.append({
                    "Model": model,
                    "Metric Value": entry["results"][selected_metric],
                    "Last Updated": entry["timestamp"]
                })
        
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values(by="Metric Value", ascending=is_ascending)
            
            # Highlight the best model
            best_model = df.iloc[0]["Model"] if not is_ascending else df.iloc[-1]["Model"]
            
            # Display leaderboard table
            st.subheader(f"Leaderboard - {selected_dataset} ({selected_metric})")
            
            # Format the metric to 4 decimal places
            df["Metric Value"] = df["Metric Value"].apply(lambda x: f"{x:.4f}")
            
            # Add ranking column
            df.insert(0, "Rank", range(1, len(df) + 1))
            
            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )
            
            # Visualize results
            st.subheader("Visual Comparison")
            
            fig = px.bar(
                df,
                x="Model",
                y="Metric Value",
                title=f"{selected_metric.capitalize()} Comparison",
                color="Model",
                height=500
            )
            
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title=selected_metric.capitalize(),
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Upload new model results
            st.subheader("Upload New Results")
            
            with st.expander("Add custom model results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    custom_model = st.text_input("Model name")
                
                with col2:
                    custom_metric = st.number_input(
                        f"{selected_metric} value",
                        min_value=0.0,
                        max_value=1.0 if selected_metric != "loss" else 10.0,
                        step=0.001
                    )
                
                if st.button("Add to Leaderboard"):
                    if custom_model and custom_metric is not None:
                        # Create results dictionary
                        custom_results = {selected_metric: float(custom_metric)}
                        
                        # Update leaderboard
                        update_leaderboard(selected_dataset, custom_model, custom_results)
                        
                        st.success(f"Added {custom_model} to the leaderboard")
                        st.experimental_rerun()
                    else:
                        st.error("Please enter both model name and metric value")
        else:
            st.info(f"No {selected_metric} data available for the selected dataset")
    else:
        st.info("No data available for the selected dataset")