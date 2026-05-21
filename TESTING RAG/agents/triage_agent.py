def get_triage_level(disease_predictions):
    """
    disease_predictions example:
    [
        {"disease": "Pneumonia", "probability": 0.78},
        {"disease": "Pneumothorax", "probability": 0.20}
    ]
    """ 

    for item in disease_predictions:
        disease = item["disease"].lower()
        prob = item["probability"]

        if "pneumothorax" in disease and prob >= 0.30:
            return {
                "level": "STAT",
                "reason": "Possible pneumothorax detected, which may require immediate medical attention."
            }

        if "pneumonia" in disease and prob >= 0.35:
            return {
                "level": "URGENT",
                "reason": "Pneumonia probability is significant and should be reviewed by a doctor soon."
            }

        if "pleural effusion" in disease and prob >= 0.40:
            return {
                "level": "URGENT",
                "reason": "Pleural effusion probability is significant and requires clinical review."
            }

    return {
        "level": "ROUTINE",
        "reason": "No high-risk abnormality crossed the urgency threshold."
    }


if __name__ == "__main__":
    sample_predictions = [
        {"disease": "Pneumonia", "probability": 0.78},
        {"disease": "Pleural effusion", "probability": 0.42}
    ]

    triage = get_triage_level(sample_predictions)

    print("\n===== TRIAGE RESULT =====\n")
    print("Level:", triage["level"])
    print("Reason:", triage["reason"])