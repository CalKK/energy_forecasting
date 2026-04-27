import sys
import os

# Test the import logic from app.py
sys.path.insert(0, os.path.abspath("src"))

try:
    from energy_modeling.pipeline import run_training, load_and_prepare
    print("SUCCESS: Successfully imported run_training and load_and_prepare")
    print(f"Imported from: {os.path.abspath('src/energy_modeling/pipeline.py')}")
except ImportError as e:
    print(f"FAILURE: {e}")
    # Print more info
    import energy_modeling
    print(f"energy_modeling location: {energy_modeling.__file__}")
    if hasattr(energy_modeling, 'pipeline'):
        print(f"energy_modeling.pipeline location: {energy_modeling.pipeline.__file__}")
        print(f"Attributes in pipeline: {dir(energy_modeling.pipeline)}")
except Exception as e:
    print(f"ERROR: {e}")
