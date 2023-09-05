from flask import Flask, render_template, request

from main import Main

app = Flask(__name__)
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
    prognosis = main.predict_disease()
    precautions = main.get_precautions()

    return render_template('form.html', prognosis=prognosis, precaution1=precautions[0], precaution2=precautions[1], precaution3=precautions[2], precaution4=precautions[3])

if __name__ == '__main__':
    app.run(debug=True)
