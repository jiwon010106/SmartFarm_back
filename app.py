from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
from images_model.chamoe_model.chamoe_model import predict_disease  # 참외 모델 예측 함수 임포트

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/weather")
async def get_weather(city: str):
    try:
        from utils.apiUrl import fetchWeatherData
        weather_data = await fetchWeatherData(city)
        
        # CORS 헤더 추가
        return weather_data
    except Exception as e:
        print(f"Weather API Error: {str(e)}")
        return {"error": str(e)}

@app.post("/predict")
async def predict_chamoe(file: UploadFile = File(...)):
    try:
        result = await predict_disease(file)
        return result
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/predictions/{crop}/{city}")
async def get_predictions(crop: str, city: str):
    try:
        from utils.apiUrl import fetchWeatherData
        
        # 작물에 따른 예측 모듈 선택
        if crop == "cabbage":
            from testpython.pricetest_v2 import predict_prices
        elif crop == "apple":
            from testpython.appleprice import predict_prices
        elif crop == "onion":  # 양파 예측 추가
            from testpython.onion import predict_prices
        else:
            raise ValueError("지원하지 않는 작물입니다")
        
        weather_data = await fetchWeatherData(city)
        predictions = predict_prices(weather_data)
        
        if 'error' in predictions:
            raise Exception(predictions['error'])
            
        return {
            "predictions": predictions,
            "weather_data": weather_data['raw']
        }
    except Exception as e:
        print(f"Error in predictions: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("Server is running on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)