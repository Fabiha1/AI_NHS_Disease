import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load training data
train_data = pd.read_csv('Training.csv')
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

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=32)

y_hat = model.predict(x_test)
y_pred = [val.argmax() for val in y_hat]  # Convert probabilities to class indices
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

model.save('tfmodel')
