from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

# ملف لحفظ الطلبات الجديدة
CSV_FILE = "requests.csv"

# الأعمدة المطلوبة
columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area']


# تحويل النصوص إلى أرقام كما في التدريب
def encode_input(data):
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    }
    data['Dependents'] = int(data['Dependents'].replace('3+', '3'))
    for key in mappings:
        data[key] = mappings[key][data[key]]
    return data


# الصفحة الرئيسية (نموذج إضافة طلب)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_data = {col: request.form[col] for col in columns}
        encoded = encode_input(form_data.copy())

        df = pd.DataFrame([encoded])
        prediction = model.predict(df)[0]

        # إضافة النتيجة
        form_data['Prediction'] = 'Approved' if prediction == 1 else 'Rejected'

        # حفظ في CSV
        df_save = pd.DataFrame([form_data])
        if os.path.exists(CSV_FILE):
            df_save.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df_save.to_csv(CSV_FILE, index=False)

        return redirect(url_for('all_requests'))

    return render_template("index.html")


# صفحة عرض كل الطلبات ونتائجها
@app.route('/requests')
def all_requests():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame()
    return render_template("all_requests.html", tables=df.to_dict(orient="records"))


# New route to display model training results
@app.route('/model_results')
def model_results():
    # --- Dummy Data for Demonstration ---
    # In a real scenario, you would pass your actual accuracy and classification report
    # from where you train/evaluate your model.
    # For now, let's use the values you provided.
    accuracy_val = 0.8288288288288288

    # This is how you'd structure the classification report data for Jinja2
    # It's a list of dictionaries, where each dictionary represents a row.
    classification_report_data = [
        {"class": "0", "precision": 0.84, "recall": 0.50, "f1_score": 0.63, "support": 32},
        {"class": "1", "precision": 0.83, "recall": 0.96, "f1_score": 0.89, "support": 79},
        {"class": "accuracy", "precision": None, "recall": None, "f1_score": 0.83, "support": 111}, # Accuracy is usually only F1-score column for overall
        {"class": "macro avg", "precision": 0.83, "recall": 0.73, "f1_score": 0.76, "support": 111},
        {"class": "weighted avg", "precision": 0.83, "recall": 0.83, "f1_score": 0.81, "support": 111},
    ]
    # --- End of Dummy Data ---

    return render_template(
        "model_results.html",
        accuracy_value=accuracy_val,
        report_data=classification_report_data
    )


@app.route('/participants')
def participants():
    project_participants = [
        "Safa_a_305107",
        "Majd_316629",
        "amir_284893",
        "Ghazala_jahjah_279160",
        "nermin_269135"
    ]
    return render_template("participants.html", participants=project_participants)



if __name__ == '__main__':
    app.run(debug=True)
