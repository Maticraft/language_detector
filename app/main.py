from fastapi import FastAPI, HTTPException

from app.model import detect_language, run_benchmark_test


app = FastAPI()


@app.post("/detect/")
def detect(text: str):
    try:
        predicted_language = detect_language(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text, "language": predicted_language}


@app.get("/benchmark/")
def benchmark():
    try:
        accuracy = run_benchmark_test()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"accuracy": accuracy}
