from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model_path = "model.pkl"  # Make sure this path is correct relative to where you run the app
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form data
    input_features = [float(x) for x in request.form.values()]
    
    # Convert to numpy array and reshape
    input_array = np.array([input_features])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_array)
    
    # Map prediction to human-readable form
    output = prediction[0]
    prediction_text = f'Predicted Injury Severity: {output}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
