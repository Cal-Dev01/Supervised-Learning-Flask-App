from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'all_data.csv')
model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scatter_plot.png')

# Ensure the uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def save_data(data):
    """Save new data to the existing data file."""
    if os.path.exists(data_file_path):
        existing_data = pd.read_csv(data_file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        updated_data = data
    updated_data.to_csv(data_file_path, index=False)


def train_model():
    """Train the model using the data from the data file."""
    data = pd.read_csv(data_file_path)
    X = data[['size', 'sound']]
    y = data['type']

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)

    # Generate and save the graph with a legend
    plt.figure()
    colors = {'dog': 'orange', 'cat': 'gray'}
    for animal_type, color in colors.items():
        subset = data[data['type'] == animal_type]
        plt.scatter(subset['size'], subset['sound'], c=color, label=animal_type)
    plt.xlabel('size')
    plt.ylabel('sound')
    plt.title('Scatter plot of Size vs Sound')
    plt.legend()
    plt.savefig(graph_path)
    plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)

        # Check if required columns are in the data
        if 'size' in data.columns and 'sound' in data.columns and 'type' in data.columns:
            save_data(data)
            train_model()
            return jsonify({'message': 'Data uploaded, model trained, and graph generated successfully!',
                            'graph_url': url_for('get_graph')})
        else:
            return jsonify({'error': 'CSV file must contain size, sound, and type columns'}), 400


@app.route('/graph')
def get_graph():
    return send_file(graph_path, mimetype='image/png')


@app.route('/predict', methods=['POST'])
def predict():
    size = request.form['size']
    sound = request.form['sound']
    data = [[float(size), float(sound)]]

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found, please upload a CSV file to train the model'}), 400

    model = joblib.load(model_path)
    prediction = model.predict(data)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
