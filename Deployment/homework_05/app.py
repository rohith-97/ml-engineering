from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Define the input data model
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI(title="Lead Scoring API", version="1.0")

# Global variable for the pipeline
pipeline = None

# Load the model at startup
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        with open('pipeline_v1.bin', 'rb') as f:
            pipeline = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
def read_root():
    return {"message": "Lead Scoring API is running!"}

@app.post("/predict")
def predict_lead_conversion(lead: LeadData):
    try:
        # Convert input to dictionary
        lead_dict = lead.dict()
        
        # Create input for prediction
        X_new = [lead_dict]
        
        # Make prediction
        prediction_proba = pipeline.predict_proba(X_new)
        conversion_probability = float(prediction_proba[0][1])
        
        return {
            "lead_source": lead.lead_source,
            "number_of_courses_viewed": lead.number_of_courses_viewed,
            "annual_income": lead.annual_income,
            "conversion_probability": conversion_probability,
            "will_convert": conversion_probability >= 0.5
        }
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# This part is only for running directly, not needed for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)