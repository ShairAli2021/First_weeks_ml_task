



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

# Calculate suggested min and max values for each feature
feature_ranges = {
    feature: {'min': X[feature].min(), 'max': X[feature].max()}
    for feature in X.columns
}

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

def get_advice(prediction):
    if prediction == 1:  # Diabetes
        return "It's important to consult with your healthcare provider for personalized advice. Maintain a balanced diet, exercise regularly, and monitor your blood sugar levels."
    else:  # No Diabetes
        return "Great news! Keep up with a healthy lifestyle, including a balanced diet and regular exercise, to maintain good overall health."

def get_suggested_values():
    return feature_ranges

@app.route('/')
def home():
    suggested_values = get_suggested_values()
    return render_template('index.html', suggested_values=suggested_values)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # ... (previous code remains unchanged)
       
        Preg = float(request.form['Preg'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        BMI = float(request.form['BMI'])
        Age = float(request.form['Age'])
        Insulin = float(request.form['Insulin'])
        
        # Validate and restrict the input for DiabetesPedigreeFunction to be between 0 and 1
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        DiabetesPedigreeFunction = max(0, min(1, DiabetesPedigreeFunction))  # Ensure it's within [0, 1]

        # Make prediction
        user_data = [[Preg,Glucose, BloodPressure, SkinThickness, BMI, Age, Insulin, DiabetesPedigreeFunction]]

        # Ensure the feature names match your dataset columns
        feature_names = ['Preg', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
        user_df = pd.DataFrame(user_data, columns=feature_names)

        # Make prediction using the trained model
        prediction = model.predict(user_df)

        # Display the result
        result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

        # Plot user input values
        plot_url = plot_user_input(user_data)
        
        advice = get_advice(prediction)

        suggested_values = get_suggested_values()

        return render_template('index.html', result=result, advice=advice, plot_url=plot_url, suggested_values=suggested_values)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
