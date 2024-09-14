Simple API for language detection tool.

The discussion of the model choice is presented in notebook 'langugage_detection_discussion.ipynb'.

The requirement for running the service is Docker.
To run the service, execute the following command:
```
docker-compose up
```
Service will be available at http://localhost:8000/, and the documentation at http://localhost:8000/docs.
To detect language of given text, send POST request to http://localhost:8000/detect with JSON body:
```
{
    "text": "text to detect language"
}
```
The response will be JSON with detected language:
```
{
    "text": "text to detect language",
    "language": "detected language"
}
```
To run the benchmark test on the Wili-2018 dataset, execute the following command:
send GET request to http://localhost:8000/benchmark