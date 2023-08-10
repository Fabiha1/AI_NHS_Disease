import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

dis_and_smpts = pd.read_csv('main.csv')

label_encoder = LabelEncoder()
dis_and_smpts['label'] = label_encoder.fit_transform(dis_and_smpts['label'])

y = dis_and_smpts['label']
x = dis_and_smpts.drop(['label', 'frequency'], axis=1)

joblib.dump(label_encoder, 'label_encoder.pkl')

# dis_and_smpts['label'] = label_encoder.fit_transform(dis_and_smpts['label'])

print(dis_and_smpts.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(x_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=750, batch_size=32)

y_hat = model.predict(x_test)
y_pred = [val.argmax() for val in y_hat]  # Convert probabilities to class indices
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

model.save('tfmodel')
