import joblib
import pandas as pd
import numpy as np
class Main:
    def __init__(self, symptoms):
        self.user_inputs = symptoms
        self.load_files()

    def load_files(self):
        # Load the saved LabelEncoder
        self.label_encoder = joblib.load('label_encoder.pkl')
        self.dis_and_smpts = pd.read_csv('Testing.csv')
        self.precautions = pd.read_csv('symptom_precaution.csv')

        # Load the trained Random Forest model
        self.loaded_model = joblib.load('random_forest_model.pkl')

    def predict_disease(self):
        symptom_positions = [self.dis_and_smpts.columns.get_loc(symptom) for symptom in self.user_inputs]

        # Initialize an array for symptoms, initializing with 0
        symptoms_array = np.zeros(len(self.dis_and_smpts.columns) - 1)

        # Set the positions of the user's symptoms to 1
        for i in symptom_positions:
            symptoms_array[i] = 1

        # Reshape the inputs to match the model's expectations
        encoded_inputs = [symptoms_array]

        # Use the loaded model for prediction
        user_pred = self.loaded_model.predict(encoded_inputs)
        user_pred_index = user_pred[0]  # Get the predicted class index

        # Convert the predicted index back to the original disease label
        predicted_disease = self.label_encoder.inverse_transform([user_pred_index])[0]

        return predicted_disease

    def get_precautions(self):
        # Find the precautions associated with the predicted disease
        predicted_precautions = self.precautions[self.precautions['Disease'] == self.predict_disease()]

        precautions =[]
        for i, row in predicted_precautions.iterrows():
            for j in range(1, 5):  # Iterate through precautions
                precaution_col = f'Precaution {j}'
                precautions.append(row[precaution_col])

        precautions = [item.capitalize() for item in precautions]
        return precautions
