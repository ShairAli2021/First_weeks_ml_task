from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the diabetes dataset
diabetes_data = pd.read_csv('data/diabetes.csv')

# Features and target
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Train a simple RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

def plot_user_input(user_data):
    # Create a bar plot for user input values
    plt.bar(X.columns, user_data[0])
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.title('User Input Values')
    
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert the plot to a base64-encoded string
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return plot_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])
        insulin = float(request.form['insulin'])
        
        # Validate and restrict the input for DiabetesPedigreeFunction to be between 0 and 1
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        DiabetesPedigreeFunction = max(0, min(1, DiabetesPedigreeFunction))  # Ensure it's within [0, 1]

        # Make prediction
        user_data = [[pregnancies, glucose, blood_pressure, skin_thickness, bmi, age, insulin, DiabetesPedigreeFunction]]

        # Ensure the feature names match your dataset columns
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
        user_df = pd.DataFrame(user_data, columns=feature_names)

        # Make prediction using the trained model
        prediction = model.predict(user_df)

        # Display the result
        result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

        # Plot user input values
        plot_url = plot_user_input(user_data)

        return render_template('index.html', result=result, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
