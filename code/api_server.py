"""FastAPI HTTP API for the Multi-Modal Evidence Review system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import process_claim
from models import ClaimInput, ClaimOutput
from evaluation.main import run_evaluation

app = FastAPI(
    title="Multi-Modal Evidence Review API",
    version="1.0.0",
    description="Verify damage claims using multi-modal evidence analysis",
)


class ProcessRequest(BaseModel):
    claim_id: str
    claim_object: str
    claimed_issue_type: str
    claimed_object_part: str
    claim_conversation: str
    user_id: str
    image_paths: list[str]


class ProcessResponse(BaseModel):
    claim_id: str
    claim_status: str
    issue_type: str
    object_part: str
    severity: str
    evidence_standard_met: str
    valid_image: str
    explanation: str
    risk_flags: list[str]


class HealthResponse(BaseModel):
    status: str
    system: str


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", system="multi-modal-evidence-review")


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    try:
        claim_input = ClaimInput(
            claim_id=req.claim_id,
            claim_object=req.claim_object,
            claimed_issue_type=req.claimed_issue_type,
            claimed_object_part=req.claimed_object_part,
            claim_conversation=req.claim_conversation,
            user_id=req.user_id,
            image_paths=req.image_paths,
        )
        output = process_claim(claim_input)
        row = output.to_csv_row()
        return ProcessResponse(
            claim_id=row["claim_id"],
            claim_status=row["claim_status"],
            issue_type=row["issue_type"],
            object_part=row["object_part"],
            severity=row["severity"],
            evidence_standard_met=row["evidence_standard_met"],
            valid_image=row["valid_image"],
            explanation=row["explanation"],
            risk_flags=[f.strip() for f in row.get("risk_flags", "").split(";") if f.strip()],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate():
    try:
        metrics = run_evaluation()
        return {"status": "ok", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
