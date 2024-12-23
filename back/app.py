from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import base64
import io

# FastAPI 애플리케이션 생성
app = FastAPI()
index_class = {}
try:
    with open("imagenet_classes.txt") as file:
        for i, line in enumerate(file):
            index_class[i] = line
except Exception as e:
    print("class-index file loading faile")
# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ResNet 모델 로드
try:
    model = models.resnet50(pretrained=True)
    model.eval()
except Exception as e:
    print("Model loading failed")

# 이미지 변환 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("File received:", file.filename)
        # 파일이 이미지인지 확인
        try:
            image = Image.open(file.file).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(content={"error": "Uploaded file is not a valid image"}, status_code=400)

        # 이미지 전처리
        transformed_image = transform(image).unsqueeze(0)  # 배치 차원 추가

        # 모델로 예측 수행
        with torch.no_grad():
            outputs = model(transformed_image)
            _, predicted = torch.max(outputs, 1)

        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 결과 반환
        return {
            "class": index_class[int(predicted.item())],
            "image": f"data:image/jpeg;base64,{image_base64}"
        }

    except Exception as e:
        return JSONResponse(content={"error": f"Unexpected error: {str(e)}"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "ResNet API is running"}
