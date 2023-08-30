import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training data
train_data = pd.read_csv('Training3.csv')

label_encoder = LabelEncoder()
train_data['prognosis'] = label_encoder.fit_transform(train_data['prognosis'])

y_train = train_data['prognosis']
x_train = train_data.drop(['prognosis'], axis=1)

# Load testing data
test_data = pd.read_csv('Testing.csv')
y_test = label_encoder.transform(test_data['prognosis'])
x_test = test_data.drop(['prognosis'], axis=1)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

print(train_data.head())

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

joblib.dump(model, 'random_forest_model.pkl')