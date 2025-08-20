from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retrieve_similar import get_similar_reports
from app.generate_report import generate_report

app = FastAPI()

# Allow frontend (React) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema
class ReportRequest(BaseModel):
    doctor_prompt: str
    patient_name: str
    birth_date: str
    dossier_number: str
    doctor_name: str
    biopsy_date: str
    

@app.post("/generate")
async def generate(request: ReportRequest):
    similar_reports = get_similar_reports(request.doctor_prompt)
    similar_contents = [r["content"] for r in similar_reports]

    generated = generate_report(
        request.doctor_prompt,
        similar_contents,
        {
            "patient_name": request.patient_name,
            "birth_date": request.birth_date,
            "dossier_number": request.dossier_number,
            "doctor_name": request.doctor_name,
            "biopsy_date": request.biopsy_date
        }
    )

    return {
        "report": generated,
        "similar_reports": [[r["id"], r["content"]] for r in similar_reports]
    }
