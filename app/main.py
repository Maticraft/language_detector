from fastapi import FastAPI, HTTPException

from app.model import detect_language, run_benchmark_test


app = FastAPI()


@app.post("/detect/")
def detect(text: str):
    '''
    Detect the language of the input text.
    '''
    try:
        predicted_language = detect_language(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"text": text, "language": predicted_language}


@app.get("/benchmark/")
def benchmark():
    '''
    Run benchmark test to evaluate the model's accuracy. It tests model accuracy on Willi-2018 dataset composed of text samples in 235 languages.
    '''
    try:
        accuracy = run_benchmark_test()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"accuracy": accuracy}
