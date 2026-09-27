import joblib
import pandas as pd


def test_performance_model_prediction():
    model = joblib.load("models/performance_model.joblib")
    preprocessor = joblib.load("models/performance_preprocessor.joblib")

    sample = pd.DataFrame([{
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 4
    }])

    processed = preprocessor.transform(sample)
    prediction = model.predict(processed)[0]

    assert 0 <= prediction <= 20