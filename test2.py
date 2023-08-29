import joblib
import pandas as pd
import numpy as np

# Load the saved LabelEncoder
label_encoder = joblib.load('label_encoder.pkl')
dis_and_smpts = pd.read_csv('Testing.csv')
precautions = pd.read_csv('symptom_precaution.csv')

# Load the trained Random Forest model
loaded_model = joblib.load('random_forest_model.pkl')

# Simulate user inputs (replace with actual user inputs)
user_inputs = ['itching', 'vomiting', 'skin_rash', 'continuous_sneezing']

symptom_positions = [dis_and_smpts.columns.get_loc(symptom) for symptom in user_inputs]

# Initialize an array for symptoms, initializing with 0
symptoms_array = np.zeros(len(dis_and_smpts.columns) - 1)

# Set the positions of the user's symptoms to 1
for i in symptom_positions:
    symptoms_array[i] = 1

# Reshape the inputs to match the model's expectations
encoded_inputs = [symptoms_array]

# Use the loaded model for prediction
user_pred = loaded_model.predict(encoded_inputs)
user_pred_index = user_pred[0]  # Get the predicted class index

# Convert the predicted index back to the original disease label
predicted_disease = label_encoder.inverse_transform([user_pred_index])[0]

print("Predicted Disease:", predicted_disease)

# Find the precautions associated with the predicted disease
predicted_precautions = precautions[precautions['Disease'] == predicted_disease]

print("\nPrecautions:")
for i, row in predicted_precautions.iterrows():
    for j in range(1, 5):  # Iterate through precautions
        precaution_col = f'Precaution_{j}'
        precaution = row[precaution_col]
        if pd.notna(precaution):
            print(f"{precaution_col}: {precaution}")
