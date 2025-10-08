# Quick Start Guide

## Running the Application

### Option 1: Using the Batch File (Easiest)
1. Double-click `run_web_app.bat`
2. Wait for the server to start
3. Open your browser to http://localhost:5000

### Option 2: Using PowerShell
```powershell
cd "d:\5th Semester\Projects\MiniProject\Try1\src"
python web_interface.py
```

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
