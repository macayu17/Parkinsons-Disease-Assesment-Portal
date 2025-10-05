class FeatureMapper:
    def __init__(self):
        # Define mapping between patient-friendly questions and dataset features
        self.feature_mapping = {
            "basic_info": {
                "age": {
                    "question": "What is your age?",
                    "type": "numeric",
                    "dataset_feature": "age"
                },
                "sex": {
                    "question": "What is your biological sex?",
                    "type": "categorical",
                    "options": ["Male", "Female"],
                    "dataset_feature": "SEX",
                    "mapping": {"Male": 1, "Female": 0}
                },
                "education": {
                    "question": "Years of education completed?",
                    "type": "numeric",
                    "dataset_feature": "EDUCYRS"
                },
                "bmi": {
                    "question": "What is your BMI? (Calculate from height and weight)",
                    "type": "numeric",
                    "dataset_feature": "BMI"
                }
            },
            "family_history": {
                "family_pd": {
                    "question": "Do you have any family members with Parkinson's Disease?",
                    "type": "categorical",
                    "options": ["No family history", "First degree relative", "Other relative"],
                    "dataset_feature": "fampd",
                    "mapping": {
                        "No family history": 3,
                        "First degree relative": 1,
                        "Other relative": 2
                    }
                }
            },
            "motor_symptoms": {
                "tremor": {
                    "question": "Have you experienced tremors or shaking?",
                    "type": "boolean",
                    "dataset_feature": "sym_tremor"
                },
                "rigidity": {
                    "question": "Do you experience stiffness or rigidity in your muscles?",
                    "type": "boolean",
                    "dataset_feature": "sym_rigid"
                },
                "bradykinesia": {
                    "question": "Have you noticed slowness in your movements?",
                    "type": "boolean",
                    "dataset_feature": "sym_brady"
                },
                "balance": {
                    "question": "Do you have issues with balance or posture?",
                    "type": "boolean",
                    "dataset_feature": "sym_posins"
                }
            },
            "non_motor_symptoms": {
                "sleep": {
                    "question": "Do you have trouble sleeping or act out your dreams?",
                    "type": "numeric",
                    "dataset_feature": "rem",
                    "scale": "0-10"
                },
                "daytime_sleepiness": {
                    "question": "How would you rate your daytime sleepiness?",
                    "type": "numeric",
                    "dataset_feature": "ess",
                    "scale": "0-10"
                },
                "mood": {
                    "question": "Have you experienced depression or anxiety?",
                    "type": "numeric",
                    "dataset_feature": "gds",
                    "scale": "0-10"
                }
            },
            "cognitive_symptoms": {
                "memory": {
                    "question": "Have you noticed any changes in your memory or thinking?",
                    "type": "numeric",
                    "dataset_feature": "moca",
                    "scale": "0-30"
                }
            }
        }
        
    def get_patient_questionnaire(self):
        """Generate a list of questions for patients"""
        questions = []
        for category in self.feature_mapping.values():
            for feature in category.values():
                questions.append({
                    "question": feature["question"],
                    "type": feature["type"],
                    "options": feature.get("options", None),
                    "scale": feature.get("scale", None)
                })
        return questions
    
    def map_patient_response_to_features(self, responses):
        """Map patient responses to dataset features"""
        feature_values = {}
        for category in self.feature_mapping.values():
            for feature_name, feature_info in category.items():
                if feature_name in responses:
                    response = responses[feature_name]
                    if "mapping" in feature_info:
                        feature_values[feature_info["dataset_feature"]] = \
                            feature_info["mapping"].get(response, None)
                    else:
                        feature_values[feature_info["dataset_feature"]] = response
        return feature_values

def main():
    # Initialize feature mapper
    mapper = FeatureMapper()
    
    # Get questionnaire
    questions = mapper.get_patient_questionnaire()
    
    # Example: Print all questions
    print("Patient Questionnaire:")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q['question']}")
        if q['options']:
            print(f"Options: {', '.join(q['options'])}")
        if q['scale']:
            print(f"Scale: {q['scale']}")

if __name__ == "__main__":
    main()