import pickle
import pandas as pd

def score_lead():
    # Define the record to score
    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }
    
    try:
        # Load the pre-trained model pipeline
        with open('pipeline_v1.bin', 'rb') as f:
            pipeline = pickle.load(f)
        
        print("Model loaded successfully!")
        
        # Convert record to the expected format
        # The pipeline might expect a list of dictionaries or specific structure
        X_new = [record]
        
        # Make prediction probability
        prediction_proba = pipeline.predict_proba(X_new)
        
        # Get probability of conversion (class 1)
        conversion_probability = prediction_proba[0][1]
        
        print(f"Probability that this lead will convert: {conversion_probability:.4f}")
        print(f"Rounded to 3 decimal places: {conversion_probability:.3f}")
        
        # Check which option matches
        options = [0.333, 0.533, 0.733, 0.933]
        closest_match = min(options, key=lambda x: abs(x - conversion_probability))
        
        print(f"\nClosest match from options: {closest_match}")
        
        return conversion_probability
        
    except Exception as e:
        print(f"Error: {e}")
        # Let's try a different approach
        return try_alternative_approach()

def try_alternative_approach():
    try:
        with open('pipeline_v1.bin', 'rb') as f:
            pipeline = pickle.load(f)
        
        print("\nTrying alternative approach...")
        
        # The pipeline might expect a DataFrame with specific column order
        record = {
            "lead_source": "paid_ads",
            "number_of_courses_viewed": 2,
            "annual_income": 79276.0
        }
        
        # Try as DataFrame
        X_new = pd.DataFrame([record])
        prediction_proba = pipeline.predict_proba(X_new)
        conversion_probability = prediction_proba[0][1]
        
        print(f"Probability that this lead will convert: {conversion_probability:.4f}")
        return conversion_probability
        
    except Exception as e:
        print(f"Alternative approach also failed: {e}")
        return None

if __name__ == "__main__":
    probability = score_lead()
    
    if probability is not None:
        print(f"\nFinal probability: {probability:.3f}")
        
        # Determine the exact match
        if abs(probability - 0.333) < 0.01:
            print("✓ The probability is 0.333")
        elif abs(probability - 0.533) < 0.01:
            print("✓ The probability is 0.533")
        elif abs(probability - 0.733) < 0.01:
            print("✓ The probability is 0.733")
        elif abs(probability - 0.933) < 0.01:
            print("✓ The probability is 0.933")
        else:
            print(f"Probability {probability:.3f} doesn't closely match any option")