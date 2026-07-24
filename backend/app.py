"""
DASS-21 Mental Health Severity Predictor — Flask REST API Backend
=================================================================
Endpoints:
  GET  /api/health          → server health check
  GET  /api/questions       → returns all 21 shuffled DASS-21 questions
  POST /api/predict         → accepts answers + demographics, returns predictions
  GET  /api/model-info      → model architecture and performance metrics
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
import json
import os
import random

app = Flask(__name__)
CORS(app)   # allow frontend (different port) to call this API

# ── Load models & metadata once at startup ────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

stress_model     = joblib.load(os.path.join(MODELS_DIR, "stress_model.pkl"))
anxiety_model    = joblib.load(os.path.join(MODELS_DIR, "anxiety_model.pkl"))
depression_model = joblib.load(os.path.join(MODELS_DIR, "depression_model.pkl"))
scaler           = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]
LABEL_MAP     = {int(k): v for k, v in metadata["label_map"].items()}
MODELS        = {
    "stress":     stress_model,
    "anxiety":    anxiety_model,
    "depression": depression_model
}

# ── All 21 DASS-21 questions with subscale tags ───────────────────────────────
# Format: (question_number, subscale, question_text, index_within_subscale)
# index_within_subscale is used to map back to stress_1..7, anxiety_1..7, depression_1..7
DASS21_QUESTIONS = [
    {"id": "q01", "number": 1,  "subscale": "stress",     "sub_index": 1,
     "text": "I found it hard to wind down"},
    {"id": "q02", "number": 2,  "subscale": "anxiety",    "sub_index": 1,
     "text": "I was aware of dryness of my mouth"},
    {"id": "q03", "number": 3,  "subscale": "depression", "sub_index": 1,
     "text": "I couldn't seem to experience any positive feeling at all"},
    {"id": "q04", "number": 4,  "subscale": "anxiety",    "sub_index": 2,
     "text": "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)"},
    {"id": "q05", "number": 5,  "subscale": "depression", "sub_index": 2,
     "text": "I found it difficult to work up the initiative to do things"},
    {"id": "q06", "number": 6,  "subscale": "stress",     "sub_index": 2,
     "text": "I tended to over-react to situations"},
    {"id": "q07", "number": 7,  "subscale": "anxiety",    "sub_index": 3,
     "text": "I experienced trembling (e.g. in the hands)"},
    {"id": "q08", "number": 8,  "subscale": "stress",     "sub_index": 3,
     "text": "I felt that I was using a lot of nervous energy"},
    {"id": "q09", "number": 9,  "subscale": "anxiety",    "sub_index": 4,
     "text": "I was worried about situations in which I might panic and make a fool of myself"},
    {"id": "q10", "number": 10, "subscale": "depression", "sub_index": 3,
     "text": "I felt that I had nothing to look forward to"},
    {"id": "q11", "number": 11, "subscale": "stress",     "sub_index": 4,
     "text": "I found myself getting agitated"},
    {"id": "q12", "number": 12, "subscale": "stress",     "sub_index": 5,
     "text": "I found it difficult to relax"},
    {"id": "q13", "number": 13, "subscale": "depression", "sub_index": 4,
     "text": "I felt sad and depressed"},
    {"id": "q14", "number": 14, "subscale": "stress",     "sub_index": 6,
     "text": "I was intolerant of anything that kept me from getting on with what I was doing"},
    {"id": "q15", "number": 15, "subscale": "anxiety",    "sub_index": 5,
     "text": "I felt I was close to panic"},
    {"id": "q16", "number": 16, "subscale": "depression", "sub_index": 5,
     "text": "I felt that I had lost interest in just about everything"},
    {"id": "q17", "number": 17, "subscale": "depression", "sub_index": 6,
     "text": "I felt I wasn't worth much as a person"},
    {"id": "q18", "number": 18, "subscale": "stress",     "sub_index": 7,
     "text": "I felt that I was rather touchy"},
    {"id": "q19", "number": 19, "subscale": "anxiety",    "sub_index": 6,
     "text": "I was aware of the action of my heart in the absence of physical exertion"},
    {"id": "q20", "number": 20, "subscale": "anxiety",    "sub_index": 7,
     "text": "I felt scared without any good reason"},
    {"id": "q21", "number": 21, "subscale": "depression", "sub_index": 7,
     "text": "I felt that life was meaningless"},
]

# ── Feedback / Suggestions per condition & severity ───────────────────────────
FEEDBACK = {
    "stress": {
        1: {
            "summary": "Your stress levels are within a healthy range.",
            "suggestions": [
                "Keep maintaining your current work-life balance.",
                "Regular physical activity helps sustain low stress levels.",
                "Practice mindfulness or meditation to stay grounded.",
                "Ensure you're getting 7-9 hours of sleep consistently."
            ],
            "warning": None
        },
        2: {
            "summary": "You are experiencing mild stress. This is common and manageable.",
            "suggestions": [
                "Try deep breathing exercises for 5-10 minutes daily.",
                "Identify your main stress triggers and address them one at a time.",
                "Take short breaks during work every 90 minutes.",
                "Consider journaling to process your thoughts."
            ],
            "warning": None
        },
        3: {
            "summary": "You are experiencing moderate stress. It's important to take action now.",
            "suggestions": [
                "Schedule dedicated relaxation time every day — even 20 minutes helps.",
                "Talk to a trusted friend or family member about what you're going through.",
                "Reduce caffeine and screen time, especially before bed.",
                "Consider speaking with a counsellor or therapist.",
                "Break large tasks into smaller manageable steps."
            ],
            "warning": "Moderate stress sustained over time can impact physical health. Please don't ignore these signs."
        },
        4: {
            "summary": "Your stress levels are severe. Please take this seriously.",
            "suggestions": [
                "Seek professional support from a psychologist or therapist as soon as possible.",
                "Inform someone you trust about how you're feeling.",
                "Avoid taking on new responsibilities until your stress reduces.",
                "Practice progressive muscle relaxation daily.",
                "Contact a mental health helpline if needed: iCall (India): 9152987821"
            ],
            "warning": "Severe stress requires professional attention. Please reach out for help."
        },
        5: {
            "summary": "Your stress levels are extremely severe. Immediate support is strongly recommended.",
            "suggestions": [
                "Please contact a mental health professional immediately.",
                "Reach out to a trusted person — you don't have to face this alone.",
                "iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7)",
                "Avoid making major life decisions while under this level of stress.",
                "Focus only on basic self-care: eating, sleeping, and breathing."
            ],
            "warning": "This level of stress is a serious health concern. Please seek help immediately."
        }
    },
    "anxiety": {
        1: {
            "summary": "Your anxiety levels are within a normal, healthy range.",
            "suggestions": [
                "Continue managing anxiety through regular exercise and routine.",
                "Limit news and social media consumption if it causes worry.",
                "Practice gratitude — write 3 things you're thankful for each day.",
                "Stay socially connected with friends and family."
            ],
            "warning": None
        },
        2: {
            "summary": "You are experiencing mild anxiety. Some worry is normal and can be managed.",
            "suggestions": [
                "Try the 4-7-8 breathing technique: inhale 4s, hold 7s, exhale 8s.",
                "Limit caffeine — it significantly worsens anxiety symptoms.",
                "Identify situations that trigger your anxiety and plan responses.",
                "Regular aerobic exercise reduces anxiety significantly."
            ],
            "warning": None
        },
        3: {
            "summary": "You are experiencing moderate anxiety. This needs attention.",
            "suggestions": [
                "Consider Cognitive Behavioural Therapy (CBT) — highly effective for anxiety.",
                "Practice grounding techniques: name 5 things you can see, 4 you can touch, 3 you can hear.",
                "Reduce alcohol — it worsens anxiety in the medium term.",
                "Speak to a doctor or mental health professional.",
                "Try to establish a consistent sleep schedule."
            ],
            "warning": "Moderate anxiety left unaddressed can worsen over time. Please seek support."
        },
        4: {
            "summary": "You are experiencing severe anxiety. Please seek professional help.",
            "suggestions": [
                "Consult a psychiatrist or psychologist — anxiety at this level responds well to treatment.",
                "Do not self-medicate with alcohol or substances.",
                "Inform someone close to you about how you're feeling.",
                "iCall (India): 9152987821 | NIMHANS helpline: 080-46110007",
                "Practice daily grounding exercises and limit stressful environments."
            ],
            "warning": "Severe anxiety is a treatable condition. Please don't delay getting professional help."
        },
        5: {
            "summary": "Your anxiety is at an extremely severe level. Please reach out for support now.",
            "suggestions": [
                "Please contact a mental health professional or go to a hospital if needed.",
                "iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7)",
                "Ask someone you trust to stay with you.",
                "Avoid being alone for extended periods.",
                "Focus on controlled breathing if you feel a panic attack coming on."
            ],
            "warning": "Extremely severe anxiety requires immediate professional attention. Please seek help today."
        }
    },
    "depression": {
        1: {
            "summary": "Your depression score is within a healthy range.",
            "suggestions": [
                "Maintain your social connections and activities you enjoy.",
                "Regular exercise, sunlight exposure, and good sleep protect against depression.",
                "Check in with yourself regularly — emotional self-awareness is protective.",
                "Stay connected with people who make you feel good."
            ],
            "warning": None
        },
        2: {
            "summary": "You are experiencing mild depression. Take these early signs seriously.",
            "suggestions": [
                "Stay physically active — even a 30-minute daily walk significantly helps.",
                "Maintain a daily routine, even on days you don't feel like it.",
                "Avoid isolating yourself — reach out to friends or family.",
                "Limit alcohol, which is a depressant.",
                "Consider speaking with a counsellor."
            ],
            "warning": None
        },
        3: {
            "summary": "You are experiencing moderate depression. Please seek support.",
            "suggestions": [
                "Speak to a doctor, psychologist, or counsellor as soon as possible.",
                "Try to do one small enjoyable activity each day, even if motivation is low.",
                "Keep a mood journal to track patterns.",
                "Avoid making big life decisions while feeling this way.",
                "iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345"
            ],
            "warning": "Moderate depression can progress if untreated. Please reach out to a professional."
        },
        4: {
            "summary": "You are experiencing severe depression. Please get help immediately.",
            "suggestions": [
                "Consult a psychiatrist or psychologist urgently.",
                "Talk to someone you completely trust about how you are feeling right now.",
                "If you are having thoughts of self-harm, please call a helpline immediately.",
                "iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7) | iCall: 9152987821",
                "Depression at this level is a medical condition and responds to treatment."
            ],
            "warning": "Severe depression is a serious medical condition. Please do not face this alone."
        },
        5: {
            "summary": "Your depression score is extremely severe. Please reach out for help right now.",
            "suggestions": [
                "Please contact a mental health professional or emergency services if needed.",
                "Vandrevala Foundation: 1860-2662-345 (24/7 free helpline)",
                "iCall: 9152987821 | NIMHANS: 080-46110007",
                "Tell someone you trust exactly how you're feeling.",
                "You are not alone and help is available. Treatment works."
            ],
            "warning": "URGENT: Please seek immediate professional support. You matter and help is available."
        }
    }
}

# ── Helper: compute features + predict ────────────────────────────────────────
def run_prediction(age, gender, marital_status, education, occupation,
                   sleep_problem, stress_responses, anxiety_responses,
                   depression_responses):
    """
    Core prediction logic — computes engineered features from raw DASS-21
    responses and demographic inputs, then runs all three ensemble models.
    """
    stress_score      = sum(stress_responses)        # 0-21
    anxiety_score     = sum(anxiety_responses)       # 0-21
    depression_score  = sum(depression_responses)    # 0-21

    cognitive_score       = sum(depression_responses[:4])   # dep items 1-4
    physiological_score   = sum(anxiety_responses[:4])      # anx items 1-4
    emotional_instability = (stress_score + anxiety_score) / 2
    comorbidity_index     = (int(stress_score > 14) +
                             int(anxiety_score > 6) +
                             int(depression_score > 8))

    raw_features = {
        "age":                    age,
        "gender":                 gender,
        "marital_status":         marital_status,
        "education":              education,
        "occupation":             occupation,
        "sleep_problem":          sleep_problem,
        "cognitive_score":        cognitive_score,
        "physiological_score":    physiological_score,
        "emotional_instability":  emotional_instability,
        "comorbidity_index":      comorbidity_index,
    }

    X_input = np.array([[raw_features[f] for f in FEATURE_NAMES]])
    X_scaled = scaler.transform(X_input)

    results = {}
    for condition, model in MODELS.items():
        pred  = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0].tolist()
        label = LABEL_MAP[pred]
        results[condition] = {
            "severity_level":  pred,
            "severity_label":  label,
            "confidence":      round(max(proba), 3),
            "probabilities":   {LABEL_MAP[i+1]: round(p, 3)
                                for i, p in enumerate(proba)},
            "raw_score":       (stress_score if condition == "stress"
                               else anxiety_score if condition == "anxiety"
                               else depression_score),
            "feedback":        FEEDBACK[condition][pred]
        }

    overall_risk = round(
        (results["stress"]["severity_level"] +
         results["anxiety"]["severity_level"] +
         results["depression"]["severity_level"]) / 3, 2
    )

    return {
        "predictions":        results,
        "overall_risk_score": overall_risk,
        "comorbidity_index":  comorbidity_index,
        "engineered_features": {
            "cognitive_score":       cognitive_score,
            "physiological_score":   physiological_score,
            "emotional_instability": emotional_instability,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms API is running and models are loaded."""
    return jsonify({
        "status":  "healthy",
        "models":  list(MODELS.keys()),
        "version": "1.0.0"
    })


@app.route("/api/questions", methods=["GET"])
def get_questions():
    """
    Returns all 21 DASS-21 questions.
    Questions are shuffled to reduce response bias.
    The subscale label is NOT sent to frontend — hidden from user.
    Query param: ?shuffle=true (default) or ?shuffle=false
    """
    shuffle = request.args.get("shuffle", "true").lower() == "true"
    questions = [
        {
            "id":     q["id"],
            "number": q["number"],   # original DASS-21 number (for reference)
            "text":   q["text"],
            # subscale intentionally hidden from public response
        }
        for q in DASS21_QUESTIONS
    ]
    if shuffle:
        random.shuffle(questions)

    response_options = [
        {"value": 0, "label": "Never",        "description": "Did not apply to me at all"},
        {"value": 1, "label": "Sometimes",    "description": "Applied to me to some degree, or some of the time"},
        {"value": 2, "label": "Often",        "description": "Applied to me to a considerable degree or a good part of the time"},
        {"value": 3, "label": "Almost Always","description": "Applied to me very much or most of the time"},
    ]

    return jsonify({
        "questions":        questions,
        "response_options": response_options,
        "instructions":     (
            "Please read each statement and select the option that best describes "
            "how much each statement applied to you over the past week."
        )
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expected JSON body:
    {
      "demographics": {
        "age": 25,
        "gender": 1,           // 0=Female, 1=Male
        "marital_status": 0,   // 0=Single, 1=Married
        "education": 3,        // 0=Illiterate,1=Primary,2=SSC,3=HSC,4=Graduation+
        "occupation": 3,       // 0=Housewife,1=Service,2=Business,3=Student,4=DayLabor,5=Unemployed
        "sleep_problem": 1     // 0=No, 1=Yes
      },
      "responses": {
        "q01": 2, "q02": 1, "q03": 1, ...  // answer for each question id (0-3)
      }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # ── Validate demographics ─────────────────────────────────────────────
        demo = data.get("demographics", {})
        required_demo = ["age", "gender", "marital_status",
                         "education", "occupation", "sleep_problem"]
        missing = [f for f in required_demo if f not in demo]
        if missing:
            return jsonify({"error": f"Missing demographic fields: {missing}"}), 400

        age            = int(demo["age"])
        gender         = int(demo["gender"])
        marital_status = int(demo["marital_status"])
        education      = int(demo["education"])
        occupation     = int(demo["occupation"])
        sleep_problem  = int(demo["sleep_problem"])

        if not (10 <= age <= 100):
            return jsonify({"error": "Age must be between 10 and 100"}), 400

        # ── Validate responses ────────────────────────────────────────────────
        responses = data.get("responses", {})
        all_ids = {q["id"] for q in DASS21_QUESTIONS}
        missing_q = [qid for qid in all_ids if qid not in responses]
        if missing_q:
            return jsonify({"error": f"Missing responses for: {missing_q}"}), 400

        invalid_q = [qid for qid, val in responses.items()
                     if int(val) not in [0, 1, 2, 3]]
        if invalid_q:
            return jsonify({"error": f"Invalid response values for: {invalid_q}. Must be 0-3"}), 400

        # ── Map responses back to ordered subscale arrays ─────────────────────
        # Build lookup: id → value
        resp_lookup = {qid: int(val) for qid, val in responses.items()}

        # Extract per subscale in sub_index order (1..7)
        stress_q_sorted     = sorted([q for q in DASS21_QUESTIONS if q["subscale"] == "stress"],
                                      key=lambda x: x["sub_index"])
        anxiety_q_sorted    = sorted([q for q in DASS21_QUESTIONS if q["subscale"] == "anxiety"],
                                      key=lambda x: x["sub_index"])
        depression_q_sorted = sorted([q for q in DASS21_QUESTIONS if q["subscale"] == "depression"],
                                      key=lambda x: x["sub_index"])

        stress_responses     = [resp_lookup[q["id"]] for q in stress_q_sorted]
        anxiety_responses    = [resp_lookup[q["id"]] for q in anxiety_q_sorted]
        depression_responses = [resp_lookup[q["id"]] for q in depression_q_sorted]

        # ── Run prediction ────────────────────────────────────────────────────
        result = run_prediction(
            age, gender, marital_status, education, occupation, sleep_problem,
            stress_responses, anxiety_responses, depression_responses
        )

        return jsonify({
            "status":  "success",
            "data":    result,
            "message": "Assessment complete. Results are for informational purposes only and do not constitute a clinical diagnosis."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Returns model architecture, performance metrics, and dataset information."""
    return jsonify({
        "model": {
            "name":        "N=3 Soft Voting Ensemble",
            "components":  ["HistGradientBoosting", "RandomForest", "MLP (α=0.01)"],
            "voting":      "soft",
            "description": "Three architecturally diverse models combined by averaging probability distributions"
        },
        "performance": metadata["model_performance"],
        "dataset": {
            "name":        "Mental Health dataset based on DASS-21",
            "source":      "Daffodil International University, Bangladesh",
            "doi":         "10.17632/br82d4xkj7.1",
            "samples":     1812,
            "year":        2024
        },
        "features": FEATURE_NAMES,
        "disclaimer": (
            "This tool is for informational and research purposes only. "
            "It does not provide a clinical diagnosis. "
            "Please consult a qualified mental health professional."
        )
    })


@app.route("/api/feedback-guide", methods=["GET"])
def feedback_guide():
    """Returns the severity level descriptions and what each level means."""
    return jsonify({
        "severity_levels": {
            "1": {"label": "Normal",           "color": "#2ecc71", "description": "Within healthy range"},
            "2": {"label": "Mild",             "color": "#f1c40f", "description": "Mild symptoms present"},
            "3": {"label": "Moderate",         "color": "#e67e22", "description": "Noticeable impact on daily life"},
            "4": {"label": "Severe",           "color": "#e74c3c", "description": "Significant impairment"},
            "5": {"label": "Extremely Severe", "color": "#8e44ad", "description": "Requires immediate professional help"}
        },
        "helplines": {
            "iCall (India)":         "9152987821",
            "Vandrevala Foundation": "1860-2662-345 (24/7)",
            "NIMHANS":               "080-46110007",
            "AASRA":                 "9820466627"
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
