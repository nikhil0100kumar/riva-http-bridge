from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile
from riva.client.asr import ASRService
import grpc
import riva.client

app = FastAPI()

RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
FUNCTION_ID = "ee8dc628-76de-4acc-8595-1836e7e857bd"
AUTH_TOKEN = "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"

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language_code: str = Form("en-US")):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        creds = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(RIVA_SERVER, creds)
        asr_service = ASRService(channel, use_ssl=True)

        riva.client.add_headers({
            "function-id": FUNCTION_ID,
            "authorization": f"Bearer {AUTH_TOKEN}"
        })

        response = asr_service.offline_recognize(tmp_path, language_code=language_code)
        return {"transcript": response}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
