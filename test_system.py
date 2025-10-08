"""
Test script to verify the Parkinson's Disease Assessment System
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import ReportGenerator, MedicalKnowledgeBase

def test_prediction():
    """Test the prediction system with sample data."""
    print("=" * 70)
    print("Testing Parkinson's Disease Assessment System")
    print("=" * 70)
    
    # Sample patient data
    sample_patient = {
        'age': 65,
        'SEX': 1,  # Male
        'EDUCYRS': 16,
        'race': 1,
        'BMI': 26.5,
        'fampd': 1,  # Positive family history
        'fampd_bin': 1,
        'sym_tremor': 2,
        'sym_rigid': 1,
        'sym_brady': 2,
        'sym_posins': 1,
        'rem': 1,
        'ess': 8,
        'gds': 3,
        'stai': 35,
        'moca': 24,
        'clockdraw': 3,
        'bjlot': 25
    }
    
    print("\nSample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    # Initialize system
    print("\n" + "=" * 70)
    print("Initializing System...")
    print("=" * 70)
    
    kb = MedicalKnowledgeBase()
    report_gen = ReportGenerator(kb, docs_dir="medical_docs")
    
    try:
        print("\nLoading models...")
        report_gen.load_models()
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        print("\nThis is expected if models haven't been trained yet.")
        print("The system will use fallback rule-based classification.")
    
    # Make prediction
    print("\n" + "=" * 70)
    print("Making Prediction...")
    print("=" * 70)
    
    try:
        prediction_results = report_gen.predict_patient(sample_patient)
        
        print("\nPrediction Results:")
        print(f"  Predicted Class: {prediction_results['ensemble_prediction']}")
        print(f"  Confidence: {prediction_results['confidence']:.2%}")
        print(f"\n  Probabilities:")
        
        class_names = ['Healthy Control', 'Parkinson\'s Disease', 'SWEDD', 'Prodromal PD']
        probs = prediction_results['ensemble_probabilities']
        
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            print(f"    {name}: {prob:.2%}")
        
        print("\n[OK] Prediction completed successfully")
        
        # Generate report
        print("\n" + "=" * 70)
        print("Generating Medical Report...")
        print("=" * 70)
        
        report = report_gen.generate_full_report(sample_patient, "TEST_PATIENT_001")
        print("\nReport Preview (first 500 characters):")
        print(report[:500] + "...")
        
        print("\n[OK] Report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    
    print("\n" + "=" * 70)
    if success:
        print("[OK] All tests passed!")
    else:
        print("[ERROR] Some tests failed. Check the error messages above.")
    print("=" * 70)
