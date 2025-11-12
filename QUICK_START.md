# Quick Start Guide

Get up and running with the Parkinson's Disease Assessment Portal in minutes!

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies
```bash
# Clone the repository
git clone https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git
cd Parkinsons-Disease-Assesment-Portal

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation (Optional)
```bash
# Test CUDA setup (if you have NVIDIA GPU)
python test_cuda.py

# Run system tests
python test_system.py
```

## 🏃 Running the Application

### Option 1: Using the Batch File (Windows - Easiest)
1. Double-click `run_web_app.bat`
2. Wait for the server to start
3. Open your browser to http://localhost:5000

### Option 2: Using Command Line
```bash
# Navigate to the project directory
cd Parkinsons-Disease-Assesment-Portal

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run the web interface
cd src
python web_interface.py
```

### Option 3: Using Python Directly
```bash
python src/web_interface.py
```

The server will start on http://localhost:5000

## Using the Assessment System

### 1. Enter Patient Data
- Go to the "Assessment" page
- Fill in required fields (marked with *)
  - Age
  - Sex
  - Education Years
  - BMI
- Fill in optional symptom data for better accuracy

### 2. Run Assessment
- Click "Validate Data" to check for errors (optional)
- Click "Run Assessment" to get predictions
- Wait for the analysis to complete

### 3. View Results
- See the predicted diagnosis
- View confidence level (High/Medium/Low)
- Check probability distribution across 4 classes:
  - Healthy Control
  - Parkinson's Disease
  - SWEDD
  - Prodromal PD

### 4. Generate Report
- Click "Generate Comprehensive Report"
- View the detailed medical report
- Download the report as a text file

## Dark Mode

Click the moon/sun icon in the top right corner to toggle between light and dark modes.

## Troubleshooting

### Server Won't Start
- Check if Python is installed: `python --version`
- Check if port 5000 is already in use
- Try running on a different port: `python web_interface.py --port 5001`

### Prediction Not Working
1. Check the terminal for error messages
2. Verify required fields are filled
3. Run the test script: `python test_system.py`

### Dark Mode Not Working
- Clear browser cache (Ctrl+F5)
- Check if JavaScript is enabled
- Try a different browser

## Keyboard Shortcuts

- **Ctrl+R** - Refresh page
- **Ctrl+F5** - Hard refresh (clear cache)
- **F12** - Open developer console (for debugging)

## Support

For technical issues, check:
1. Terminal output for error messages
2. Browser console (F12) for JavaScript errors
3. FIXES_SUMMARY.md for detailed technical information
