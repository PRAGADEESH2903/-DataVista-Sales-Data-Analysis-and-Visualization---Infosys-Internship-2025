# E-commerce Delivery Analytics Dashboard

This web application provides analytics and predictions for e-commerce delivery performance, focusing on delivery delays and refund requests.

## Features

- Interactive dashboard with key metrics
- Delivery delay and refund request predictions
- Visualizations of delivery performance
- Platform and product category analysis

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `models` directory and place your trained models there:
```bash
mkdir models
# Place your trained models (delay_model.pkl and refund_model.pkl) in the models directory
```

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
  - `index.html`: Home page with prediction form
  - `dashboard.html`: Analytics dashboard
- `models/`: Directory for trained models
- `requirements.txt`: Python dependencies

## Data Analysis

The dashboard provides insights into:
- Total number of orders
- Average delivery time
- Delivery delay rates
- Refund request rates
- Platform-wise performance
- Product category analysis

## Prediction Model

The prediction model takes into account:
- Platform
- Product category
- Order value
- Expected delivery time

## License

MIT License 