from flask import Flask, render_template, request

from main import Main

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'stank'
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = []
    for i in range(1, 7):
        symptom_input = request.form.get(f'symptomInput{i}')
        if symptom_input:
            symptoms.append(symptom_input)

    main = Main(symptoms)
    print(main.predict_disease())

    return "Data Received Successfully"

if __name__ == '__main__':
    app.run(debug=True)
