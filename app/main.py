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
    Run benchmark test to evaluate the model's accuracy. It tests model accuracy on Willi-2018 dataset composed of text samples in 235 languages. It may take a few minutes to complete.

    '''
    try:
        accuracy = run_benchmark_test()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"accuracy": accuracy}


@app.get("/benchmark/{languages_subset}")
def benchmark(languages_subset: str):
    '''
    Run benchmark test to evaluate the model's accuracy on the subset of languages. It tests model accuracy on Willi-2018 dataset composed of text samples in 235 languages.
    To evaluate the model on a subset of languages, pass a list of language codes as a query parameter.
    Example codes for 20 languages evaluated in discussion paper:
        Catalan - cat
        Czech - ces
        Danish - dan
        German - deu
        English - eng
        Spanish - spa
        Estonian - est
        Finnish - fin
        French - fra
        Croatian - hrv
        Hungarian - hun
        Italian - ita
        Lithuanian - lit
        Dutch - nld
        Norwegian - nob
        Polish - pol
        Portuguese - por
        Romanian - ron
        Swedish - swe
        Turkish - tur
    Example: /benchmark/cat,ces,dan,deu,eng,spa,est,fin,fra,hrv,hun,ita,lit,nld,nob,pol,por,ron,swe,tur
    '''
    try:
        languages_subset = languages_subset.split(",")
        accuracy = run_benchmark_test(languages_subset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"accuracy": accuracy}
