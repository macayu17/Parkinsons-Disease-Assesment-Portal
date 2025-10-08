"""
Start the Parkinson's Disease Assessment Web Application
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Parkinson's Disease Assessment Portal")
print("=" * 70)
print()

# Check dependencies
print("Checking dependencies...")
try:
    import flask
    print("[OK] Flask installed")
except ImportError:
    print("[ERROR] Flask not installed. Run: pip install flask")
    sys.exit(1)

try:
    import pandas
    print("[OK] Pandas installed")
except ImportError:
    print("[ERROR] Pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import numpy
    print("[OK] NumPy installed")
except ImportError:
    print("[ERROR] NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import sklearn
    print("[OK] Scikit-learn installed")
except ImportError:
    print("[ERROR] Scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

print()
print("All dependencies found!")
print()

# Change to src directory
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
os.chdir(src_dir)

print("=" * 70)
print("Starting Web Server...")
print("=" * 70)
print()
print("The application will be available at:")
print("  → http://localhost:5000")
print()
print("Press Ctrl+C to stop the server")
print("=" * 70)
print()

# Import and run the app
from web_interface import app, initialize_system

# Initialize the system
print("Initializing AI models...")
if initialize_system():
    print("[OK] System initialized successfully")
    print()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
else:
    print("[WARNING] System initialization failed")
    print("The app will still run but may use fallback predictions")
    print()
    
    # Run anyway
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
