from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import uvicorn
import io

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
downy_mildew_model = load_model('images_model/chamoe_model/downy.model.keras')  # 노균병 모델
powdery_mildew_model = load_model('images_model/chamoe_model/powdery_model.keras')  # 흰가루병 모델

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # 이미지 읽기 및 전처리
        contents = await file.read()
        img = image.load_img(io.BytesIO(contents), target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 예측 수행
        downy_prob = float(downy_mildew_model.predict(img_array)[0][0])
        powdery_prob = float(powdery_mildew_model.predict(img_array)[0][0])

        # 결과 판정
        if downy_prob < 0.5 and powdery_prob < 0.5:
            result = "정상"
        else:
            result = "노균병" if downy_prob > powdery_prob else "흰가루병"

        return {
            "success": True,
            "result": result,
            "downy_probability": downy_prob,
            "powdery_probability": powdery_prob
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

