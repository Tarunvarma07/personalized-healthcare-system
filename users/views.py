
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

import seaborn as sns
from django.core.files.storage import FileSystemStorage
from django.db import IntegrityError
from django.utils import timezone

def UserRegisterActions(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        loginid = request.POST.get('loginid')
        password = request.POST.get('password')
        mobile = request.POST.get('mobile')
        email = request.POST.get('email')
        locality = request.POST.get('locality')
        status = request.POST.get('status', 'waiting')  # default to 'waiting'

        try:
            # Create user manually
            user = UserRegistrationModel.objects.create(
                name=name,
                loginid=loginid,
                password=password,
                mobile=mobile,
                email=email,
                locality=locality,
                status=status,
                date_joined=timezone.now()
            )
            user.save()
            messages.success(request, '✅ You have been successfully registered.')

        except IntegrityError as e:
            if 'email' in str(e).lower():
                messages.error(request, '❌ Email already exists.')
            elif 'mobile' in str(e).lower():
                messages.error(request, '❌ Mobile number already exists.')
            elif 'loginid' in str(e).lower():
                messages.error(request, '❌ Login ID already exists.')
            else:
                messages.error(request, f'❌ Registration failed: {str(e)}')

    return render(request, 'UserRegistrations.html')


from django.contrib import messages
from django.shortcuts import render, redirect
from .models import UserRegistrationModel

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get("loginid")
        password = request.POST.get("pswd")
        print("Login ID:", loginid)
        print("Password:", password)

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=password)
            status = user.status.lower()

            if status == "activated":
                # Set session variables
                request.session['id'] = user.id
                request.session['loginid'] = user.loginid
                request.session['password'] = user.password
                request.session['email'] = user.email
                return render(request, 'users/UserHome.html')

            elif status == "waiting":
                messages.warning(request, "⚠️ Your account is waiting for admin approval.")
            elif status == "blocked":
                messages.error(request, "🚫 Your account has been blocked by the admin.")
            else:
                messages.info(request, f"Account status: {status}")

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "❌ Invalid login credentials.")

    return render(request, 'UserLogin.html')



from .models import PredictionHistory
from django.utils.timesince import timesince

def UserHome(request):
    user_id = request.session.get('id')
    user = UserRegistrationModel.objects.get(id=user_id)

    # Count of predictions made by user
    prediction_count = PredictionHistory.objects.filter(user=user).count()

    # Recent predictions (latest 3)
    recent_predictions = PredictionHistory.objects.filter(user=user).order_by('-created_at')[:3]

    # Dummy model accuracy (or load from model training)
    model_accuracy = 90.0

    # Dummy health alerts (you can connect to alerts model later)
    health_alerts = 2

    prediction_logs = []
    for p in recent_predictions:
        prediction_logs.append({
            'disease': p.predicted_disease,
            'confidence': round(p.confidence, 1),
            'time': timesince(p.created_at) + " ago",
        })

    return render(request, "users/UserHome.html", {
        'prediction_count': prediction_count,
        'model_accuracy': model_accuracy,
        'health_alerts': health_alerts,
        'prediction_logs': prediction_logs,
    })



def view_data(request):
    from django.conf import settings
    import pandas as pd
    import os

    file_path = os.path.join(settings.MEDIA_ROOT, 'final_dataset_30000.csv')
    d = pd.read_csv(file_path)

    # Move 'Disease' column to the end if it exists
    if 'Disease' in d.columns:
        cols = [col for col in d.columns if col != 'Disease'] + ['Disease']
        d = d[cols]

    # Show only first 100 records
    d = d.head(100)

    context = {'dataset': d}
    return render(request, 'users/dataset.html', context)



# Django View for Model Training


from django.shortcuts import render
import numpy as np
import joblib
import google.generativeai as genai
from .models import PredictionHistory

# Load ML components once at the top (recommended)
best_model = joblib.load("media/best_model.pkl")
scaler = joblib.load("media/scaler.pkl")
label_encoders = joblib.load("media/label_encoders.pkl")
disease_encoder = joblib.load("media/disease_encoder.pkl")

# Configure Gemini API for Generative AI predictions
API_KEY = 'AIzaSyCRbSKELoZT7P1mPn0Ikftr82IsgZTBSpw'
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-3-flash-preview')

# Disease to precautions mapping (ML-based)
DISEASE_PRECAUTIONS = {
    # Common diseases and their precautions
    'Common Cold': [
        'Rest and drink plenty of fluids',
        'Use over-the-counter cold medications for symptom relief',
        'Gargle with warm salt water for sore throat',
        'Use a humidifier to ease congestion',
        'Avoid close contact with others to prevent spreading'
    ],
    'Flu': [
        'Get plenty of rest and stay hydrated',
        'Take antiviral drugs if prescribed by a doctor',
        'Use pain relievers for fever and aches',
        'Stay home to avoid spreading the virus',
        'Cover coughs and sneezes with a tissue'
    ],
    'Diabetes': [
        'Monitor blood sugar levels regularly',
        'Follow a healthy, balanced diet',
        'Exercise regularly as recommended by your doctor',
        'Take medications as prescribed',
        'Check feet daily for any wounds or infections'
    ],
    'Hypertension': [
        'Monitor blood pressure regularly',
        'Reduce salt intake in your diet',
        'Exercise regularly (30 minutes most days)',
        'Maintain a healthy weight',
        'Limit alcohol consumption and avoid smoking'
    ],
    'Asthma': [
        'Avoid known triggers and allergens',
        'Use prescribed inhalers correctly',
        'Monitor breathing and peak flow regularly',
        'Keep rescue medications available',
        'Get regular check-ups with your doctor'
    ],
    'Arthritis': [
        'Maintain a healthy weight to reduce joint stress',
        'Exercise regularly with low-impact activities',
        'Use heat or cold therapy for pain relief',
        'Take medications as prescribed',
        'Protect joints during daily activities'
    ],
    'Migraine': [
        'Identify and avoid personal triggers',
        'Practice relaxation techniques and stress management',
        'Get adequate sleep and maintain regular sleep patterns',
        'Stay hydrated and eat regular meals',
        'Use prescribed medications at the first sign of symptoms'
    ],
    'Heart Disease': [
        'Follow a heart-healthy diet low in saturated fats',
        'Exercise regularly as recommended by your doctor',
        'Take prescribed medications consistently',
        'Monitor and control blood pressure and cholesterol',
        'Quit smoking and limit alcohol consumption'
    ],
    'Depression': [
        'Follow the treatment plan prescribed by your mental health professional',
        'Stay connected with supportive friends and family',
        'Engage in regular physical activity',
        'Practice good sleep hygiene',
        'Avoid alcohol and recreational drugs'
    ],
    'Anxiety': [
        'Practice relaxation techniques like deep breathing',
        'Engage in regular physical exercise',
        'Limit caffeine and alcohol intake',
        'Maintain a consistent sleep schedule',
        'Seek professional help and follow treatment plans'
    ]
}
import pandas as pd
import numpy as np
import joblib
import os

from django.shortcuts import render
from django.conf import settings

from .models import UserRegistrationModel, PredictionHistory
from .train_models import train_models


# =========================
# SAFE MODEL LOADER
# =========================
def load_ml_objects():

    base = "media"

    model = joblib.load(os.path.join(base, "best_model.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler.pkl"))
    disease_encoder = joblib.load(os.path.join(base, "disease_encoder.pkl"))
    gender_encoder = joblib.load(os.path.join(base, "gender_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(base, "feature_columns.pkl"))

    return model, scaler, disease_encoder, gender_encoder, feature_columns


# =========================
# PREDICTION VIEW
# =========================
def prediction(request):
    predicted_disease = None
    precautions = None
    confidence = None

    if request.method == "POST":
        try:
            # Load ML objects ONLY when needed
            model, scaler, disease_encoder, gender_encoder, feature_columns = load_ml_objects()

            age = int(request.POST.get("age"))
            gender = request.POST.get("gender")

            symptoms = [
                request.POST.get(f"symptom_{i+1}")
                for i in range(7)
            ]

            data = {"Age": age, "Gender": gender}

            for i in range(7):
                data[f"Symptom{i+1}"] = symptoms[i]

            input_df = pd.DataFrame([data])

            # encode gender
            input_df["Gender"] = gender_encoder.transform(input_df["Gender"])

            # onehot symptoms
            symptom_cols = [f"Symptom{i}" for i in range(1, 8)]
            input_df = pd.get_dummies(input_df, columns=symptom_cols)

            # match training features
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            # scale
            input_scaled = scaler.transform(input_df)

            # predict
            pred_idx = model.predict(input_scaled)[0]
            predicted_disease = disease_encoder.inverse_transform([pred_idx])[0]

            confidence = float(model.predict_proba(input_scaled).max() * 100)

            # save history
            user_id = request.session.get('id')
            user = UserRegistrationModel.objects.get(id=user_id)

            PredictionHistory.objects.create(
                user=user,
                predicted_disease=predicted_disease,
                confidence=confidence
            )

            precautions_list = DISEASE_PRECAUTIONS.get(predicted_disease, [
                'Consult doctor',
                'Maintain healthy lifestyle',
                'Follow treatment plan'
            ])

            precautions = "\n• " + "\n• ".join(precautions_list)

        except Exception as e:
            predicted_disease = "Error in prediction"
            precautions = str(e)

    return render(request, "users/prediction.html", {
        "predicted_disease": predicted_disease,
        "precautions": precautions,
        "confidence": confidence
    })


# =========================
# TRAINING VIEW
# =========================
def training(request):
    accuracies, best_model_name = train_models()

    return render(request, 'users/modelresults.html', {
        'accuracies': accuracies,
        'best_model': best_model_name
    })
def generative_ai_prediction(request):
    predicted_disease = None
    analysis = None

    if request.method == "POST":
        try:
            # Get age & gender
            age = request.POST.get("age")
            gender = request.POST.get("gender")

            # Collect symptoms
            symptoms = []
            for i in range(1, 6):
                symptom = request.POST.get(f'symptom_{i}')
                if symptom:
                    symptoms.append(symptom)

            if symptoms and age and gender:

                symptoms_list = ", ".join(symptoms)

                prompt = (
                    f"Patient details:\n"
                    f"Age: {age}\n"
                    f"Gender: {gender}\n\n"
                    f"Symptoms: {symptoms_list}\n\n"
                    f"Provide:\n"
                    f"1. Most likely disease\n"
                    f"2. Causes\n"
                    f"3. Precautions\n"
                    f"4. When to see doctor\n"
                    f"5. Health advice\n"
                    f"Use bullet points.\n"
                    f"Add disclaimer at end."
                )

                response = gemini_model.generate_content(prompt)
                analysis = response.text

                # Disease extraction
                text = response.text.lower()
                if "most likely" in text:
                    start = text.find("most likely") + len("most likely")
                    end = text.find(".", start)
                    if end != -1:
                        predicted_disease = text[start:end].strip().capitalize()

            else:
                analysis = "Please enter age, gender and at least one symptom."

        except Exception as e:
            predicted_disease = "Error"
            analysis = str(e)

    return render(request, "users/generative_prediction.html", {
        "predicted_disease": predicted_disease,
        "analysis": analysis
    })

def chatbot_question(request):
    if request.method == "POST":
        try:
            import json
            data = json.loads(request.body)
            question = data.get('question', '')

            if question:
                # Create prompt for health-related questions
                prompt = (
                    f"Answer this health-related question: {question}\n\n"
                    f"Provide a detailed, informative response including:\n"
                    f"1. Direct answer to the question\n"
                    f"2. Explanation and relevant medical information\n"
                    f"3. Practical advice or recommendations\n"
                    f"4. When to seek professional medical help\n\n"
                    f"Format the response clearly with proper sections.\n"
                    f"Add this disclaimer at the end: 'Disclaimer: I am GWmini AI, an AI health assistant. "
                    f"This information is for educational purposes only and should not replace professional medical advice. "
                    f"Always consult with a qualified healthcare provider for medical concerns.'"
                )

                # Get response from Gemini
                response = gemini_model.generate_content(prompt)
                answer = response.text

                return JsonResponse({'answer': answer})
            else:
                return JsonResponse({'answer': 'Please ask a health-related question.'}, status=400)

        except Exception as e:
            return JsonResponse({'answer': f"Sorry, I encountered an error: {str(e)}"}, status=500)

    return JsonResponse({'answer': 'Invalid request method.'}, status=405)