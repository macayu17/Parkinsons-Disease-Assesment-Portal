"""
Web Interface for Parkinson's Disease Assessment System.
Flask-based web application for patient data input and automated report generation.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import json
from datetime import datetime
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from rag_system import ReportGenerator, MedicalKnowledgeBase
from document_manager import DocumentManager

# Set template folder to the templates directory in the project root
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'parkinson_assessment_secret_key_2024'

# Initialize global components
report_generator = None
knowledge_base = MedicalKnowledgeBase()
document_manager = DocumentManager("medical_docs")

def initialize_system():
    """Initialize the ML models and report generator."""
    global report_generator
    try:
        # Initialize document manager with medical documents
        print(f"Loaded {document_manager.get_document_count()} medical documents")
        
        # Initialize report generator with document manager
        report_generator = ReportGenerator(knowledge_base, docs_dir="medical_docs")
        report_generator.load_models()
        print("System initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False

@app.route('/')
def index():
    """Main page with patient assessment form."""
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    """Patient assessment form page."""
    return render_template('assessment.html')

@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html', knowledge_base=knowledge_base)

@app.route('/documents')
def documents():
    """Document management page."""
    docs = document_manager.get_all_documents()
    return render_template('documents.html', documents=docs)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Handle document upload."""
    try:
        file = request.files['document']
        title = request.form['title']
        author = request.form['author']
        
        if file and title:
            filename = file.filename
            file_path = os.path.join('medical_docs', filename)
            file.save(file_path)
            
            # Add document to document manager
            document_manager.add_document(file_path, title=title, author=author)
            
            flash('Document uploaded successfully!', 'success')
        else:
            flash('Please provide a file and title', 'danger')
            
    except Exception as e:
        flash(f'Error uploading document: {str(e)}', 'danger')
        
    return redirect(url_for('documents'))

@app.route('/delete_document/<doc_id>', methods=['POST'])
def delete_document(doc_id):
    """Delete a document."""
    try:
        document_manager.remove_document(doc_id)
        flash('Document deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')
        
    return redirect(url_for('documents'))

@app.route('/view_document/<doc_id>')
def view_document(doc_id):
    """View a document."""
    doc = document_manager.get_document(doc_id)
    if doc:
        return render_template('view_document.html', document=doc)
    else:
        flash('Document not found', 'danger')
        return redirect(url_for('documents'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        # Get patient data from request
        patient_data = request.json
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'SEX', 'EDUCYRS', 'BMI']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Initialize system if not already done
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Make prediction
        prediction_results = report_generator.predict_patient(patient_data)
        
        # Map prediction to class name
        class_names = ['Healthy Control', 'Parkinson\'s Disease', 'SWEDD', 'Prodromal PD']
        predicted_class = class_names[prediction_results['ensemble_prediction']]
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': float(prediction_results['confidence']),
            'probabilities': {
                'Healthy Control': float(prediction_results['ensemble_probabilities'][0]),
                'Parkinson\'s Disease': float(prediction_results['ensemble_probabilities'][1]),
                'SWEDD': float(prediction_results['ensemble_probabilities'][2]),
                'Prodromal PD': float(prediction_results['ensemble_probabilities'][3])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """API endpoint for generating comprehensive medical reports."""
    try:
        # Get patient data from request
        data = request.json
        patient_data = data.get('patient_data', {})
        patient_id = data.get('patient_id', None)
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Initialize system if not already done
        if report_generator is None:
            if not initialize_system():
                return jsonify({'error': 'System initialization failed'}), 500
        
        # Generate report
        report = report_generator.generate_full_report(patient_data, patient_id)
        
        # Save report
        filename = f"report_{patient_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = report_generator.save_report(report, filename)
        
        response = {
            'report': report,
            'filename': filename,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Report generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_report/<filename>')
def download_report(filename):
    """Download generated report file."""
    try:
        filepath = os.path.join('reports', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Report file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_data', methods=['POST'])
def validate_data():
    """Validate patient data before processing."""
    try:
        patient_data = request.json
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Age validation
        age = patient_data.get('age')
        if age is not None:
            if age < 18 or age > 100:
                validation_results['errors'].append('Age must be between 18 and 100')
                validation_results['valid'] = False
            elif age > 80:
                validation_results['warnings'].append('Advanced age may affect assessment accuracy')
        
        # BMI validation
        bmi = patient_data.get('BMI')
        if bmi is not None:
            if bmi < 15 or bmi > 50:
                validation_results['errors'].append('BMI must be between 15 and 50')
                validation_results['valid'] = False
        
        # MoCA score validation
        moca = patient_data.get('moca')
        if moca is not None:
            if moca < 0 or moca > 30:
                validation_results['errors'].append('MoCA score must be between 0 and 30')
                validation_results['valid'] = False
        
        # Symptom scores validation (typically 0-4 scale)
        symptom_fields = ['sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins']
        for field in symptom_fields:
            value = patient_data.get(field)
            if value is not None and (value < 0 or value > 4):
                validation_results['errors'].append(f'{field} must be between 0 and 4')
                validation_results['valid'] = False
        
        return jsonify(validation_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def system_status():
    """Get system status and model information."""
    try:
        status = {
            'system_initialized': report_generator is not None,
            'models_loaded': False,
            'timestamp': datetime.now().isoformat()
        }
        
        if report_generator is not None:
            status['models_loaded'] = (
                report_generator.ensemble is not None and 
                report_generator.preprocessor is not None
            )
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Parkinson's Disease Assessment Web Interface...")
    print("Initializing ML models...")
    
    # Initialize system on startup
    if initialize_system():
        print("System ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize system. Please check model files.")