# Insurance Premium API

A RESTful API for programmatically accessing the premium prediction model. Built with FastAPI.

## Features

- **Prediction Endpoint**: Submit health data and get a premium estimate.
- **Input Validation**: Strict schema validation using Pydantic.
- **Confidence Intervals**: Returns upper and lower bounds for estimates.

## Quick Start

### 1. Requirements

Ensure you have the dependencies installed:

```bash
pip install -r ../requirements.txt
# plus fastapi uvicorn if not already there, though ideally add to requirements.txt
pip install fastapi uvicorn
```

### 2. Run the API

From the project root:

```bash
uvicorn api.main:app --reload
```

### 3. Usage

Open your browser to `http://127.0.0.1:8000/docs` to see the Swagger UI and test endpoint interactively.

**Example Request:**

```json
POST /predict
{
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
```

**Example Response:**

```json
{
  "premium_price": 15000.0,
  "currency": "EUR/USD/INR",
  "confidence_interval": {
    "lower": 5200.0,
    "upper": 24800.0
  }
}
```
