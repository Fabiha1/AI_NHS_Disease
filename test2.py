import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved LabelEncoder
label_encoder = joblib.load('label_encoder.pkl')
dis_and_smpts = pd.read_csv('main.csv')

# Load the trained model
loaded_model = load_model('tfmodel')

# Simulate user inputs (replace with actual user inputs)
user_inputs = ['shortness of breath', 'palpitation', 'pain chest', 'asthenia', 'dizziness', 'sweat', 'nausea', 'fall', 'sweating increased', 'pressure chest', 'vertigo']

symptom_positions = [dis_and_smpts.columns.get_loc(symptom) for symptom in user_inputs]

# Initialize an array for symptoms, initializing with 0
symptoms_array = np.zeros(len(dis_and_smpts.columns[1:])-1)

# Map symptom names to their corresponding positions in the encoder
for i in symptom_positions:
    symptoms_array[i] = 1

# Reshape the inputs to match the model's expectations
encoded_inputs = [symptoms_array]

# Reshape the inputs to match the model's expectations
encoded_inputs = np.array(encoded_inputs)  # Convert the list to a numpy array
encoded_inputs = np.reshape(encoded_inputs, (len(encoded_inputs), -1))  # Reshape to (batch_size, num_features)

# Use the loaded model for prediction
user_pred_probs = loaded_model.predict(encoded_inputs)
user_pred_index = user_pred_probs.argmax(axis=1)[0]  # Get the index of the predicted class

# Convert the predicted index back to the original disease label
predicted_disease = label_encoder.inverse_transform([user_pred_index])[0]

print("Predicted Disease:", predicted_disease)
