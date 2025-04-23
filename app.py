from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
import os
import json

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Load the combined model and preprocessing objects
model_path = os.path.join(os.path.dirname(__file__), 'combined_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    try:
        # Load the data
        df = pd.read_csv('Ecommerce_Delivery_Analytics_New.csv')
        
        # Calculate key metrics
        total_orders = len(df)
        avg_delivery_time = df['Delivery Time (Minutes)'].mean()
        delay_rate = (df['Delivery Delay'].sum() / total_orders) * 100
        refund_rate = (df['Refund Requested'].sum() / total_orders) * 100
        avg_rating = df['Service Rating'].mean()
        
        # Platform-wise metrics
        platform_metrics = df.groupby('Platform').agg({
            'Delivery Time (Minutes)': 'mean',
            'Delivery Delay': lambda x: (x.sum() / len(x)) * 100,
            'Refund Requested': lambda x: (x.sum() / len(x)) * 100,
            'Service Rating': 'mean'
        }).round(2).to_dict('index')
        
        # Category-wise metrics
        category_metrics = df.groupby('Product Category').agg({
            'Delivery Time (Minutes)': 'mean',
            'Delivery Delay': lambda x: (x.sum() / len(x)) * 100,
            'Refund Requested': lambda x: (x.sum() / len(x)) * 100,
            'Service Rating': 'mean'
        }).round(2).to_dict('index')
        
        # Create delivery time distribution plot
        delivery_plot = {
            'data': [{
                'x': df['Delivery Time (Minutes)'],
                'type': 'histogram',
                'name': 'Delivery Time',
                'marker': {'color': '#3498db'},
                'nbinsx': 30
            }],
            'layout': {
                'title': 'Delivery Time Distribution',
                'xaxis': {'title': 'Delivery Time (Minutes)'},
                'yaxis': {'title': 'Count'},
                'showlegend': False
            }
        }
        
        # Create platform performance plot
        platform_plot = {
            'data': [{
                'x': df['Platform'].value_counts().index,
                'y': df['Platform'].value_counts().values,
                'type': 'bar',
                'name': 'Orders',
                'marker': {'color': '#2ecc71'}
            }],
            'layout': {
                'title': 'Orders by Platform',
                'xaxis': {'title': 'Platform'},
                'yaxis': {'title': 'Number of Orders'},
                'showlegend': False
            }
        }
        
        # Create category distribution pie chart
        category_pie = {
            'data': [{
                'values': df['Product Category'].value_counts().values,
                'labels': df['Product Category'].value_counts().index,
                'type': 'pie',
                'name': 'Categories',
                'marker': {'colors': ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6']}
            }],
            'layout': {
                'title': 'Product Category Distribution',
                'showlegend': True
            }
        }
        
        # Create platform market share pie chart
        platform_pie = {
            'data': [{
                'values': df['Platform'].value_counts().values,
                'labels': df['Platform'].value_counts().index,
                'type': 'pie',
                'name': 'Platforms',
                'marker': {'colors': ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']}
            }],
            'layout': {
                'title': 'Platform Market Share',
                'showlegend': True
            }
        }
        
        # Create box plot for delivery time by category
        box_plot = {
            'data': [{
                'y': df[df['Product Category'] == category]['Delivery Time (Minutes)'],
                'type': 'box',
                'name': category,
                'boxpoints': 'outliers'
            } for category in df['Product Category'].unique()],
            'layout': {
                'title': 'Delivery Time by Category',
                'xaxis': {'title': 'Product Category'},
                'yaxis': {'title': 'Delivery Time (Minutes)'},
                'showlegend': False
            }
        }
        
        # Create service rating trend plot
        line_plot = {
            'data': [{
                'x': df['Service Rating'].value_counts().sort_index().index,
                'y': df['Service Rating'].value_counts().sort_index().values,
                'type': 'line',
                'name': 'Rating Distribution',
                'marker': {'color': '#f1c40f'}
            }],
            'layout': {
                'title': 'Service Rating Distribution',
                'xaxis': {'title': 'Rating'},
                'yaxis': {'title': 'Count'},
                'showlegend': False
            }
        }
        
        # Try to load prediction history
        try:
            with open('prediction_history.json', 'r') as f:
                prediction_history = json.load(f)
                latest_prediction = prediction_history[-1] if prediction_history else None
        except (FileNotFoundError, json.JSONDecodeError):
            latest_prediction = None
        
        # Create comparison plots if there's a latest prediction
        if latest_prediction:
            # Delivery time comparison
            delivery_comparison = {
                'data': [
                    {
                        'x': ['Average', 'Your Prediction'],
                        'y': [avg_delivery_time, latest_prediction['delivery_time']],
                        'type': 'bar',
                        'name': 'Delivery Time',
                        'marker': {'color': ['#3498db', '#e74c3c']}
                    }
                ],
                'layout': {
                    'title': 'Delivery Time Comparison',
                    'yaxis': {'title': 'Minutes'},
                    'showlegend': False
                }
            }
            
            # Rating comparison
            rating_comparison = {
                'data': [
                    {
                        'x': ['Average', 'Your Prediction'],
                        'y': [avg_rating, latest_prediction['rating_prediction']],
                        'type': 'bar',
                        'name': 'Rating',
                        'marker': {'color': ['#3498db', '#e74c3c']}
                    }
                ],
                'layout': {
                    'title': 'Rating Comparison',
                    'yaxis': {'title': 'Rating'},
                    'showlegend': False
                }
            }
            
            # Risk comparison
            risk_comparison = {
                'data': [
                    {
                        'x': ['Average', 'Your Prediction'],
                        'y': [delay_rate, latest_prediction['delay_probability'] * 100],
                        'type': 'bar',
                        'name': 'Delay Risk',
                        'marker': {'color': ['#3498db', '#e74c3c']}
                    }
                ],
                'layout': {
                    'title': 'Delay Risk Comparison',
                    'yaxis': {'title': 'Percentage'},
                    'showlegend': False
                }
            }
        else:
            delivery_comparison = None
            rating_comparison = None
            risk_comparison = None
        
        return render_template('dashboard.html',
                            total_orders=total_orders,
                            avg_delivery_time=avg_delivery_time,
                            delay_rate=delay_rate,
                            refund_rate=refund_rate,
                            avg_rating=avg_rating,
                            platform_metrics=platform_metrics,
                            category_metrics=category_metrics,
                            delivery_plot=json.dumps(delivery_plot),
                            platform_plot=json.dumps(platform_plot),
                            category_pie=json.dumps(category_pie),
                            box_plot=json.dumps(box_plot),
                            line_plot=json.dumps(line_plot),
                            platform_pie=json.dumps(platform_pie),
                            latest_prediction=latest_prediction,
                            delivery_comparison=json.dumps(delivery_comparison) if delivery_comparison else None,
                            rating_comparison=json.dumps(rating_comparison) if rating_comparison else None,
                            risk_comparison=json.dumps(risk_comparison) if risk_comparison else None)
    
    except Exception as e:
        print(f"Error in dashboard route: {str(e)}")
        # Return default values when an error occurs
        return render_template('dashboard.html',
                            error=str(e),
                            total_orders=0,
                            avg_delivery_time=0,
                            delay_rate=0,
                            refund_rate=0,
                            avg_rating=0,
                            platform_metrics={},
                            category_metrics={},
                            delivery_plot=json.dumps({}),
                            platform_plot=json.dumps({}),
                            category_pie=json.dumps({}),
                            box_plot=json.dumps({}),
                            line_plot=json.dumps({}),
                            platform_pie=json.dumps({}),
                            latest_prediction=None,
                            delivery_comparison=None,
                            rating_comparison=None,
                            risk_comparison=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['platform', 'product_category', 'order_value', 'delivery_time']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate field types
        try:
            platform = str(data['platform'])
            product_category = str(data['product_category'])
            order_value = float(data['order_value'])
            delivery_time = float(data['delivery_time'])
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid data type: {str(e)}'}), 400
        
        # Create feature vector with all required columns
        features = pd.DataFrame({
            'Platform': [platform],
            'Product Category': [product_category],
            'Order Value (INR)': [order_value],
            'Delivery Time (Minutes)': [delivery_time],
            'Order ID': ['PRED0001'],
            'Customer ID': ['PREDCUST'],
            'Order Date & Time': ['00:00.0'],
            'Customer Feedback': ['Prediction'],
            'Service Rating': [5]
        })
        
        # Make predictions using the combined model
        try:
            predictions = model.predict_proba(features)
            
            # Check if predictions array has the expected shape
            if len(predictions) == 0:
                return jsonify({'error': 'No predictions returned from model'}), 500
                
            # The model returns probabilities for each class in sequence
            # For binary classification, we get [P(No), P(Yes)] for each target
            delay_prob = predictions[0][1]  # Probability of delay (Yes)
            refund_prob = predictions[0][3]  # Probability of refund (Yes)
            
            # Convert probabilities to Yes/No based on 1% threshold
            delay_prediction = "Yes" if delay_prob > 0.01 else "No"
            refund_prediction = "Yes" if refund_prob > 0.01 else "No"
            
            # For rating, we'll use a more sophisticated calculation based on multiple factors
            base_rating = 5.0
            
            # Adjust rating based on delivery time (penalize longer delivery times)
            delivery_penalty = max(0, (delivery_time - 30) / 60)  # Penalty starts after 30 minutes
            base_rating -= min(2, delivery_penalty)  # Max 2 point penalty for delivery time
            
            # Adjust rating based on order value (higher value orders expect better service)
            value_bonus = min(1, order_value / 2000)  # Bonus up to 1 point for high-value orders
            base_rating += value_bonus
            
            # Adjust rating based on platform (historical performance)
            platform_ratings = {
                'JioMart': 0.2,
                'Blinkit': 0.1,
                'BigBasket': 0.3
            }
            base_rating += platform_ratings.get(platform, 0)
            
            # Adjust rating based on product category
            category_ratings = {
                'Fruits & Vegetables': 0.1,
                'Dairy': 0.2,
                'Beverages': 0.1
            }
            base_rating += category_ratings.get(product_category, 0)
            
            # Ensure rating is between 1 and 5
            rating = min(5, max(1, round(base_rating)))
            
            # Calculate confidence based on how close the rating is to the extremes
            rating_prob = 0.7 + (0.3 * (1 - abs(rating - 3) / 2))  # Higher confidence for extreme ratings
            
            return jsonify({
                'delay_prediction': delay_prediction,
                'refund_prediction': refund_prediction,
                'rating_prediction': rating,
                'delay_probability': float(delay_prob),
                'refund_probability': float(refund_prob),
                'rating_probability': float(rating_prob)
            })
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 