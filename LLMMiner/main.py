from fastapi import FastAPI

app = FastAPI(title="MXLLM Research Server")

@app.get("/")
async def health_check():
    return {"status": "OK"}