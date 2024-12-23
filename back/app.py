from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import base64
import io

app = FastAPI()
index_class = {}
try:
    with open("imagenet_classes.txt") as file:
        for i, line in enumerate(file):
            index_class[i] = line
    print("File Loaded")
except Exception as e:
    print("class-index file loading failed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = models.resnet50(pretrained=True)
    model.eval()
    print("Model Loaded")
except Exception as e:
    print("Model loading failed")

try:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Transformation Loaded")
except Exception as e:
    print("Transformation failed")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("File received:", file.filename)
        try:
            image = Image.open(file.file).convert("RGB")
            print("Image opened")
        except UnidentifiedImageError:
            return JSONResponse(content={"error": "Uploaded file is not a valid image"}, status_code=400)

        try:
            transformed_image = transform(image).unsqueeze(0)  
            print("Transformed")
        except Exception as e:
            print("Transform failed")

        try:
            with torch.no_grad():
                outputs = model(transformed_image)
                _, predicted = torch.max(outputs, 1)
            print(f"{int(predicted)} Predict Success")
        except Exception as e:
            print("Prediction error")

        try:
            image = Image.open(file.file).convert("RGB")
            image = image.resize((128, 128))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=50)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print("Image saved")

            image_path = f"static/{file.filename}"
            image.save(image_path, format="JPEG")
        except Exception as e:
            print("Image uploading failed")
        return {
            "class": index_class[int(predicted.item())],
            "image_base64": f"data:image/jpeg;base64,{image_base64}"
            #"image_url": f"https://resnet-back.onrender.com/{image_path}"
        }

    except Exception as e:
        return JSONResponse(content={"error": f"Unexpected error: {str(e)}"}, status_code=500)

@app.get("/")
async def root():
    return {"message": "ResNet API is running"}
