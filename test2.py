import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

dis_and_smpts = pd.read_csv('testdata.csv')
# Assuming 'dis_and_smpts' is your DataFrame containing the new data
new_data = pd.get_dummies(dis_and_smpts)

# Load the trained model (if not already loaded)
model = load_model('tfmodel')  # Replace 'your_model_file.h5' with the actual filename of your trained model

# For multi-class classification:
y_probabilities = model.predict(new_data)
y_predictions = [val.argmax() for val in y_probabilities]

# For binary classification (using a threshold of 0.5):
# y_probabilities = model.predict(new_data)
# y_predictions = [0 if val < 0.5 else 1 for val in y_probabilities]

# Convert class indices back to original class labels
label_encoder = LabelEncoder()
label_encoder.fit(dis_and_smpts['label'])
predicted_labels = label_encoder.inverse_transform(y_predictions)

print("Predicted Labels:", predicted_labels)
