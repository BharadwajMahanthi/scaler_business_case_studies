from pydantic import BaseModel, Field, validator

class InsuranceInput(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Age of the primary beneficiary")
    Diabetes: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    BloodPressureProblems: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    AnyTransplants: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    AnyChronicDiseases: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    Height: int = Field(..., ge=100, le=250, description="Height in cm")
    Weight: int = Field(..., ge=30, le=200, description="Weight in kg")
    KnownAllergies: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    HistoryOfCancerInFamily: int = Field(..., ge=0, le=1, description="0 for No, 1 for Yes")
    NumberOfMajorSurgeries: int = Field(..., ge=0, le=3, description="Number of major surgeries")

    class Config:
        schema_extra = {
            "example": {
                "Age": 30,
                "Diabetes": 0,
                "BloodPressureProblems": 0,
                "AnyTransplants": 0,
                "AnyChronicDiseases": 0,
                "Height": 175,
                "Weight": 70,
                "KnownAllergies": 0,
                "HistoryOfCancerInFamily": 0,
                "NumberOfMajorSurgeries": 0
            }
        }

class PredictionOutput(BaseModel):
    premium_price: float
    currency: str = "INR" # Or USD based on context, dataset implies ~25k which fits INR or USD depending on dataset origin. Assuming generic currency derived from dataset context.
    confidence_interval: dict
