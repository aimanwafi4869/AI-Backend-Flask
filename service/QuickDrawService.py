
import base64
import io
import subprocess
import cv2
from flask import jsonify, request
import keras
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import pickle
import torch
from flask import Flask, request, jsonify
from PIL import Image as PILImage
from PIL.Image import open as pil_open
from PIL import ImageDraw as PILImageDraw
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import TFAutoModelForImageClassification
import traceback
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# os.environ["ULTRALYTICS_CACHE"] = "False"   # Global cache kill (once per script)

class QuickDrawService:

    def __init__(self):
        # print('init')
        self.existingModel() # <--- Cannot be used in debug mode
        pass

    def createModel(self):
        self.model = YOLO("yolo11n.pt")
        
    def downloadLabels(self):
        # labels_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
        # self.labels = np.loadtxt(labels_url, dtype=str, delimiter="\n", encoding="utf-8")
        
        with open('D:\\Project\\Jogja\\AI-Backend-Flask\\model\\quickdraw\\categories.txt', "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        np.save("model/quickdraw/labels.npy", lines)

    def existingModel(self):
        try:
            self.labels = np.load("model/quickdraw/labels.npy") 
            # print(self.labels)
            # print("Labels loaded:", len(self.labels))  # Should be 345
            # print("First 5 labels:", self.labels[:5])  # Expect: ['airplane', 'apple', 'banana', 'baseball bat', 'basketball']
            # print("Index of 'airplane':", self.labels.index('airplane'))  # Should be 0
        except:
            self.downloadLabels()
            self.labels = np.load("model/quickdraw/labels.npy") 
            # print(self.labels)
        # print(self.labels)
        self.model = keras.saving.load_model("D:\\Project\\Jogja\\AI-Backend-Flask\\model\\quickdraw\\quickdraw_classifier.keras")
    
    def preprocess_sketch(self,image):
        background = PILImage.new('RGB', image.size, (255, 255, 255))  # white background
        background.paste(image, mask=image.split()[-1])             # use alpha as mask
        image = background
        image = image.resize((28, 28), PILImage.Resampling.LANCZOS).convert('L')
        
        # plt.subplot(1, 3, 1)
        # plt.title("Original ")
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        arr = np.array(image, dtype=np.float32)
        arr = 255.0 - arr          
        
        arr = arr / 255.0
        
        # plt.subplot(1, 3, 2)
        # plt.title("Original ")
        # plt.imshow(arr)
        # plt.axis('off')
        # plt.show()
        # print(arr.shape)
        arr = arr.reshape(1, 28, 28, 1)
        return arr
    
    def detect(self):
        # if "file" not in request.files:
        #     return jsonify({"error": "No file"}), 400

        # file = request.json['image']
        file = request.files['file']
        # print(file)
        if file == None:
            return {"error": "No file"}
        try:
            
            # original_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
            img = pil_open(io.BytesIO(file.read()))
            # current_image = cv2.imread('images/current_images.png')
            # image_array = np.array(img)
            # plt.subplot(1, 3, 1)
            # plt.title("Original ")
            # plt.imshow(image_array)
            # plt.axis('off')
            # # plt.show()
            # black = [0, 0, 0]
            # white = [255, 255, 255]

            # # Create a mask for black pixels
            # # np.where returns the indices where the condition is true
            # black_pixels_mask = np.where(current_image == black, white, axis=-1)
            # plt.subplot(1, 3, 2)
            # plt.title("Chnaged")
            # plt.imshow(black_pixels_mask)
            # plt.axis('off')
            # plt.show()
            img = self.preprocess_sketch(img)
            # image_array /= 255.0
            # print(image_array.dtype)
            # print(image_array.shape)
            # print(image_array)
            # image_data = base64.b64decode(file)
            # image_bytes = io.BytesIO(image_data)
            # image_bytes = io.BytesIO(image_data)
            # cv2.imwrite('images/my_image.png', image_data)
            # with open("images/imageToSave.png", "wb") as fh:
            #     fh.write(base64.decodebytes(image_data))
            
            # img = PILImage.open(image_bytes).convert('L')
            # plt.figure(figsize=(8, 4))
            # plt.subplot(1, 3, 1)
            # plt.title("Original Image")
            # plt.imshow(mpimg.imread(image_bytes, format='png'))   # or 'jpg' if it's JPEG
            # plt.axis('off')
            # plt.show()
            # nparr = np.frombuffer(image_data , np.uint8)
            # image_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            # success = cv2.imwrite('images/my_image.png', image_np)
            # # np.save(image_bytes, original_array)
            # # nparr = np.load(image_bytes)# Original resized image
            # plt.subplot(1, 3, 1)
            # plt.title("Original Resized")
            # plt.imshow(image_array, cmap='gray')
            # plt.axis('off')

            # # After your binary threshold
            # plt.subplot(1, 3, 2)
            # plt.title("After Threshold")
            # plt.imshow(img_binary, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            # plt.subplot(1, 3, 2)
            # plt.title("After Threshold")
            # plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')
            # print(image_array.shape)
            # _, img_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # # img_padded = np.pad(img_binary, ((2,2), (2,2)), mode='constant', constant_values=0)
            # # img_final  = img_padded[2:30, 2:30]
            # img_ready = img_binary.astype(np.float32) / 255.0
            # # img_ready = np.expand_dims(img_ready, axis=-1)
            # # img_ready = np.expand_dims(img_ready, axis=0)
            # # np.set_printoptions(threshold=np.inf, linewidth=120, suppress=True)
            # image_array = image_array.reshape((1,28,28,4))
            # # plt.figure(figsize=(8, 4))
            # _, image_array = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
            # image_array = image_array.astype('float32') / 255.0
            # image_array = image_array.reshape((1,28,28,1))
            

            # # # FINAL image actually sent to the model (with invisible padding trick)
            # plt.subplot(1, 3, 3)
            # # plt.title(f"FINAL → Model Input\nPredicted: {predicted_class}")
            # plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
            # plt.axis('off')

            # # plt.tight_layout()
            # plt.show()

            # plt.figure(figsize=(8, 4))

            # plt.subplot(1, 3, 2)
            # plt.title("After convert to np array")
            # plt.imshow(img_ready)
            # plt.axis('off')

            # x = self.preprocess_sketch(image_np)
            probs = self.model.predict(img, verbose=0)[0]
            idx = int(np.argmax(probs))
            return jsonify({
                "predicted_class": str(self.labels[idx]),
                "confidence": float(probs[idx]),
                "top_predictions": [
                    {"class": str(self.labels[i]), "confidence": float(probs[i])}
                    for i in np.argsort(probs)[-3:][::-1]
                ]
            })
        except Exception as e:
            e.with_traceback(None)
            print()
            return jsonify({"error": str(e)}), 500
        
    def testBlank(self):
        blank = np.ones((28, 28), dtype=np.float32)
        x = np.expand_dims(blank, axis=0)
        probs = self.model.predict(x)[0]
        print("Blank probs max:", np.max(probs), "argmax:", np.argmax(probs))
        # Test simple line (e.g., for "line" or "zigzag")
        img = PILImage.new('L', (28, 28), 255)
        draw = PILImageDraw.Draw(img)
        draw.line((0,0, 28,28), fill=0, width=2)
        arr = np.array(img) / 255.0
        x = np.expand_dims(arr, axis=0)
        probs = self.model.predict(x)[0]
        idx = np.argmax(probs)
        print("Predicted:", self.labels[idx], "Conf:", probs[idx])
        return jsonify({
                "predicted_class": str(self.labels[idx]),
                "confidence": float(probs[idx]),
                "top_predictions": [
                    {"class": str(self.labels[i]), "confidence": float(probs[i])}
                    for i in np.argsort(probs)[-3:][::-1]
                ]
            })
    
# class QuickDrawService:

#     def __init__(self):
#         print('init')

        
#     def downloadLabels(self):
#         # labels_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
#         # self.labels = np.loadtxt(labels_url, dtype=str, delimiter="\n", encoding="utf-8")
        
#         with open('D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw\\categories.txt', "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]
#         np.save("model/quickdraw/labels.npy", lines)

#     def existingModel(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         try:
#             self.labels = np.load("model/quickdraw/labels.npy") 
#         except:
#             self.downloadLabels()
#             self.labels = np.load("model/quickdraw/labels.npy") 

#         # self.model = snapshot_download(repo_id="WinKawaks/SketchXAI-Tiny-QuickDraw345", 
#         #           local_dir='D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw', 
#         #           local_dir_use_symlinks=False)
        
#         self.processor = AutoImageProcessor.from_pretrained('D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw')
#         self.model = AutoModelForImageClassification.from_pretrained('D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw')
#         self.model.to(self.device)
#         self.model.eval()  # Inference mode
        
#     def preprocess(self,image_byte):
#         # 1. Open image
#         image = pil_open(io.BytesIO(image_byte))

#         # 2. Ensure grayscale
#         if image.mode != "L":
#             image = image.convert("L")

#         # 3. Resize to 28x28
#         image = image.resize((28, 28))

#         # 4. Convert to 3-channel (RGB) - REQUIRED for processor
#         image_rgb = PILImage.new("RGB", image.size)
#         for i in range(3):
#             image_rgb.paste(image, (0, 0))

#         # 5. Let processor handle normalization + tensor
#         inputs = self.processor(image_rgb, return_tensors="pt")
#         if not inputs or 'pixel_values' not in inputs:
#             raise ValueError("Processor failed to return pixel_values")

#         return {k: v.to(self.device) for k, v in inputs.items()}
    
#     def detect(self):
#         if "file" not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files["file"]
#         if not file.filename:
#             return jsonify({"error": "Empty filename"}), 400

#         try:
#             img_bytes = file.read()
#             inputs = self.preprocess(img_bytes)
            
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

#             idx = int(np.argmax(probs))

#             return {
#                 "predicted_class": str(self.labels[idx]),
#                 "confidence": float(probs[idx]),
#                 "top_predictions": [
#                     {"class": str(self.labels[i]), "confidence": float(probs[i])}
#                     for i in np.argsort(probs)[-3:][::-1]
#                 ]
#             }
#         except Exception as e:
#             return {"error": f"Prediction failed: {str(e)}"}

# class QuickDrawService:

#     def __init__(self):
#         print('init')

#     def createModel(self):
#         self.model = YOLO("yolo11n.pt")
        
#     def downloadLabels(self):
#         # labels_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
#         # self.labels = np.loadtxt(labels_url, dtype=str, delimiter="\n", encoding="utf-8")
        
#         with open('D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw\\categories.txt', "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]
#         np.save("model/quickdraw/labels.npy", lines)

#     def existingModel(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         try:
#             self.labels = np.load("model/quickdraw/labels.npy") 
#             # print(self.labels)
#             print("Labels loaded:", len(self.labels))  # Should be 345
#             print("First 5 labels:", self.labels[:5])  # Expect: ['airplane', 'apple', 'banana', 'baseball bat', 'basketball']
#             print("Index of 'airplane':", self.labels.index('airplane'))  # Should be 0
#         except:
#             self.downloadLabels()
#             self.labels = np.load("model/quickdraw/labels.npy") 
#             # print(self.labels)
#         self.processor = AutoImageProcessor.from_pretrained(
#             'JoshuaKelleyDs/quickdraw-ConvNeXT-Tiny-Finetune')
#         self.model = AutoModelForImageClassification.from_pretrained('JoshuaKelleyDs/quickdraw-ConvNeXT-Tiny-Finetune').to(self.device)
#         self.model.eval()
    
#     def preprocess_sketch(self,image):
#         # 1. Convert to grayscale
#         if image.mode != "L":
#             image = image.convert("L")

#         # 2. Resize
#         image = image.resize((224, 224), PILImage.Resampling.LANCZOS)

#         # 3. Replicate to 3 channels (RGB)
#         # image_rgb = PILImage.merge("RGB", [image] * 3)

#         # 4. Process with 3-channel config
#         inputs = self.processor(
#             images=image,
#             return_tensors="pt"
#         )

#         return inputs["pixel_values"].to(self.device)  # [1, 3, 224, 224]
    
#     def detect(self):
#         if "file" not in request.files:
#             return jsonify({"error": "No file"}), 400
#         file = request.files["file"]
#         try:
#             img = PILImage.open(io.BytesIO(file.read()))
#             x = self.preprocess_sketch(img)  # [1, 1, 224, 224]

#             with torch.no_grad():
#                 outputs = self.model(x)
#                 logits = outputs.logits
#                 probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

#             idx = int(probs.argmax())
#             top3_idx = np.argsort(probs)[-3:][::-1]

#             return jsonify({
#                 "predicted_class": self.model.labels[idx],
#                 "confidence": float(probs[idx]),
#                 "top_predictions": [
#                     {"class": self.model.labels[i], "confidence": float(probs[i])}
#                     for i in top3_idx
#                 ]
#             })
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"error": str(e)}), 500
        
#     def testBlank(self):
#         blank = np.ones((28, 28), dtype=np.float32)
#         x = np.expand_dims(blank, axis=0)
#         probs = self.model.predict(x)[0]
#         print("Blank probs max:", np.max(probs), "argmax:", np.argmax(probs))
#         # Test simple line (e.g., for "line" or "zigzag")
#         img = PILImage.new('L', (28, 28), 255)
#         draw = PILImageDraw.Draw(img)
#         draw.line((0,0, 28,28), fill=0, width=2)
#         arr = np.array(img) / 255.0
#         x = np.expand_dims(arr, axis=0)
#         probs = self.model.predict(x)[0]
#         idx = np.argmax(probs)
#         print("Predicted:", self.labels[idx], "Conf:", probs[idx])
#         return jsonify({
#                 "predicted_class": str(self.labels[idx]),
#                 "confidence": float(probs[idx]),
#                 "top_predictions": [
#                     {"class": str(self.labels[i]), "confidence": float(probs[i])}
#                     for i in np.argsort(probs)[-3:][::-1]
#                 ]
#             })

# class QuickDrawService:

#     def __init__(self):
#         print('init')

#     def createModel(self):
#         self.model = YOLO("yolo11n.pt")
        
#     def downloadLabels(self):
#         # labels_url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
#         # self.labels = np.loadtxt(labels_url, dtype=str, delimiter="\n", encoding="utf-8")
        
#         with open('D:\\Project\\Computer Vision\\FlaskApp-main\\model\\quickdraw\\categories.txt', "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]
#         np.save("model/quickdraw/labels.npy", lines)

#     def existingModel(self):
#         try:
#             self.labels = np.load("model/quickdraw/labels.npy") 
#             print(self.labels)
#             print("Labels loaded:", len(self.labels))  # Should be 345
#             print("First 5 labels:", self.labels[:5])  # Expect: ['airplane', 'apple', 'banana', 'baseball bat', 'basketball']
#             print("Index of 'airplane':", self.labels.index('airplane'))  # Should be 0
#         except:
#             self.downloadLabels()
#             self.labels = np.load("model/quickdraw/labels.npy") 
#             print(self.labels)

#         self.model = keras.applications.MobileNetV2(
#             input_shape=(224, 224, 3),
#             include_top=False,
#             weights='imagenet'
#         )
#         self.model.trainable = False
#         self.model = keras.Sequential([
#             keras.layers.Resizing(224, 224),
#             keras.layers.Rescaling(1./255),
#             self.model,
#             keras.layers.GlobalAveragePooling2D(),
#             keras.layers.Dense(345, activation='softmax')
#         ])
    
#     def strokes_to_image(self, strokes, size=28):
#         img = np.zeros((size, size), dtype=np.uint8)
#         x, y = 0, 0
#         for dx, dy, p1, p2, p3 in strokes:
#             x += dx
#             y += dy
#             if p1 > 0.5 and 0 <= int(x) < size and 0 <= int(y) < size:
#                 img[int(y), int(x)] = 255
#             if p3 > 0.5: break
#         # Convert to 3-channel
#         img = np.stack([img]*3, axis=-1)
#         return img

#     def detect(self):
#         data = request.get_json()
#         strokes = data.get("strokes")
#         if not strokes:
#             return jsonify({"error": "missing strokes"}), 400

#         img = self.strokes_to_image(strokes)
#         img_batch = np.expand_dims(img, 0)
#         preds = self.model.predict(img_batch, verbose=0)[0]
#         top5_idx = preds.argsort()[-5:][::-1]

#         result = [
#             {"label": self.labels[i], "confidence": float(preds[i])}
#             for i in top5_idx
#         ]
#         return jsonify({"top5": result})
        
#     def testBlank(self):
#         blank = np.ones((28, 28), dtype=np.float32)
#         x = np.expand_dims(blank, axis=0)
#         probs = self.model.predict(x)[0]
#         print("Blank probs max:", np.max(probs), "argmax:", np.argmax(probs))
#         # Test simple line (e.g., for "line" or "zigzag")
#         img = PILImage.new('L', (28, 28), 255)
#         draw = PILImageDraw.Draw(img)
#         draw.line((0,0, 28,28), fill=0, width=2)
#         arr = np.array(img) / 255.0
#         x = np.expand_dims(arr, axis=0)
#         probs = self.model.predict(x)[0]
#         idx = np.argmax(probs)
#         print("Predicted:", self.labels[idx], "Conf:", probs[idx])
#         return jsonify({
#                 "predicted_class": str(self.labels[idx]),
#                 "confidence": float(probs[idx]),
#                 "top_predictions": [
#                     {"class": str(self.labels[i]), "confidence": float(probs[i])}
#                     for i in np.argsort(probs)[-3:][::-1]
#                 ]
#             })
    