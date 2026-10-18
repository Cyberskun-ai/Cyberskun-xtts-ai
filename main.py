from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
import threading
import os
import uvicorn

app = FastAPI(title="Chai Vichea AI Voice Cloning API")

# អនុញ្ញាតឱ្យ Web Browser ភ្ជាប់មក API បាន (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_status = {"ready": False, "message": "កំពុងទាញយក និងផ្ទុកម៉ូដែល AI XTTS v2..."}
tts = None

def load_ai_model():
    global tts, model_status
    try:
        print("🚀 កំពុងផ្ទុកម៉ូដែល XTTS v2...")
        # អាចប្រើប្រាស់ "cuda" ប្រសិនបើមាន GPU ឬ "cpu" សម្រាប់ CPU
        device = "cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        model_status["ready"] = True
        model_status["message"] = "ម៉ូដែល AI រួចរាល់សម្រាប់ដំណើរការ!"
        print("✅ ម៉ូដែល XTTS v2 ត្រូវបានផ្ទុកជោគជ័យ!")
    except Exception as e:
        model_status["message"] = f"មានបញ្ហាក្នុងការផ្ទុកម៉ូដែល: {str(e)}"

# ផ្ទុកម៉ូដែលក្នុង Background Thread ដើម្បីការពារការកកស្ទះ
threading.Thread(target=load_ai_model, daemon=True).start()

@app.get("/api/status")
def get_status():
    return model_status

@app.post("/api/clone-voice")
async def clone_voice(
    text: str = Form(...),
    language: str = Form("en"),
    speaker_file: UploadFile = File(...)
):
    if not model_status["ready"]:
        raise HTTPException(status_code=400, detail="ម៉ូដែល AI កំពុងទាញយក ឬមិនទាន់រួចរាល់!")

    # រក្សាទុក File សំឡេងគំរូដែល បាន Upload ចូល
    temp_speaker_path = "temp_speaker.wav"
    try:
        contents = await speaker_file.read()
        with open(temp_speaker_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"មិនអាចរក្សាទុក File សំឡេងបានឡើយ: {str(e)}")

    output_path = "output_cloned.wav"

    try:
        # ដំណើរការសំយោគ និង Clone សំឡេងតាមរយៈ XTTS v2
        tts.tts_to_file(
            text=text,
            speaker_wav=temp_speaker_path,
            language=language,
            file_path=output_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"មានបញ្ហាក្នុងការ Clone សំឡេង: {str(e)}")

    return FileResponse(output_path, media_type="audio/wav", filename="cloned_voice.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)