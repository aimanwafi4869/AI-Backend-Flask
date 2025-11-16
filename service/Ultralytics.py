
import base64
import subprocess
import cv2
from flask import jsonify, request
import numpy as np
from ultralytics import YOLO
import os
import pickle
import torch
# os.environ["ULTRALYTICS_CACHE"] = "False"   # Global cache kill (once per script)
class Ultralytics:

    def __init__(self):
        print('init')

    def createModel(self):
        self.model = YOLO("yolo11n.pt")
    
    def existingModel(self):
        self.model = YOLO('D:\\Project\\Computer Vision\\FlaskApp-main\\Yolo_v11_Model.pt')
        
    def modelPerformance(self):
        self.existingModel()
        # self.model.eval()
        results = self.model.val(
            data="D:\\Project\\Computer Vision\\FlaskApp-main\\dataset\\data.yaml",
            imgsz=640,
            batch=16,
            device="0",          # change to "cpu" if no GPU
            save_json=True,
            save_txt=True,
            plots=True,
        )
        # Print clean results
        print("\nVALIDATION RESULTS (Emotion Recognition)")
        print("="*60)
        metrics = results.results_dict
        print(f"Precision (P):  {metrics['metrics/precision(B)']:.4f}")
        print(f"Recall    (R):  {metrics['metrics/recall(B)']:.4f}")
        print(f"mAP@0.5:        {metrics['metrics/mAP50(B)']:.4f}")
        print(f"mAP@0.5:0.95:   {metrics['metrics/mAP50-95(B)']:.4f}")
        print(f"Speed (ms/img): {results.speed['inference']:.2f}")
        return {}

    def train(self):
        self.createModel()
        trained_model = self.model.train(
            data="D:\\Project\\Computer Vision\\FlaskApp-main\\dataset\\data.yaml",  # path to dataset YAML
            epochs=100,
            imgsz=640,
            batch=84,                    # PERFECT FOR RTX 3080 10GB → 9.1GB VRAM used
            device=0,
            workers=3,                   # Don't overload CPU
            optimizer="AdamW",
            lr0=0.01,
            amp=True,                    # Mixed precision = +50% speed, -40% VRAM
            cache="ram",                 # Use RAM cache (fast SSD/NVMe)
            close_mosaic=10,
            freeze=10,                   # Freeze first 10 layers = faster + less VRAM
            project="model",
            name="yolo4",
            exist_ok=True,
            patience=20,
            pretrained=True,
            seed=42
        )

        export_formats = [
            "onnx",      # Web + Streamlit
            "tflite",    # Android (realme, vivo, Samsung MY)
            "engine",    # TensorRT (RTX 3060/4070 - FASTEST!)
            "openvino",  # Intel CPU (Acer, Lenovo MY)
            "coreml",    # iPhone/iPad
            "paddle",    # China deployment
            "saved_model" # TensorFlow web
        ]

        for fmt in export_formats:
            try:
                exported = trained_model.export(
                    format=fmt,
                    imgsz=640,
                    half=True,         # FP16 = faster on mobile
                    batch=1,
                    device=0,
                    simplify=True,
                    dynamic=False
                )
                print(f"{fmt.upper():12} → {exported}")
            except Exception as e:
                print(f"{fmt.upper():12} → FAILED ({e})")

        # Move all exported files to MY folder
        import shutil
        for file in os.listdir("."):
            if file.startswith("best") and not file.endswith(".pt"):
                shutil.move(file, os.path.join("D:\\Project\\Computer Vision\\FlaskApp-main\\model", file))

    def retrain(self):
        self.existingModel()
        trained_model = self.model.train(
            data="D:\\Project\\Computer Vision\\FlaskApp-main\\dataset\\data.yaml",  # path to dataset YAML
            epochs=100,
            imgsz=640,
            batch=84,                    # PERFECT FOR RTX 3080 10GB → 9.1GB VRAM used
            device=0,
            workers=3,                   # Don't overload CPU
            optimizer="AdamW",
            lr0=0.01,
            amp=True,                    # Mixed precision = +50% speed, -40% VRAM
            cache="ram",                 # Use RAM cache (fast SSD/NVMe)
            close_mosaic=10,
            freeze=10,                   # Freeze first 10 layers = faster + less VRAM
            project="model",
            name="yolo11.1",
            exist_ok=True,
            patience=20,
            pretrained=True,
            seed=42,
        )

        export_formats = [
            "onnx",      # Web + Streamlit
            "tflite",    # Android (realme, vivo, Samsung MY)
            "engine",    # TensorRT (RTX 3060/4070 - FASTEST!)
            "openvino",  # Intel CPU (Acer, Lenovo MY)
            "coreml",    # iPhone/iPad
            "paddle",    # China deployment
            "saved_model" # TensorFlow web
        ]

        for fmt in export_formats:
            try:
                exported = trained_model.export(
                    format=fmt,
                    imgsz=640,
                    half=True,         # FP16 = faster on mobile
                    batch=1,
                    device=0,
                    simplify=True,
                    dynamic=False
                )
                print(f"{fmt.upper():12} → {exported}")
            except Exception as e:
                print(f"{fmt.upper():12} → FAILED ({e})")

        # Move all exported files to MY folder
        import shutil
        for file in os.listdir("."):
            if file.startswith("best") and not file.endswith(".pt"):
                shutil.move(file, os.path.join("D:\\Project\\Computer Vision\\FlaskApp-main\\model", file))

    def exportModel(self):
        self.existingModel()
        export_formats = [
            "onnx",      # Web + Streamlit
            # "tflite",    # Android (realme, vivo, Samsung MY)
            # "engine",    # TensorRT (RTX 3060/4070 - FASTEST!)
            # "openvino",  # Intel CPU (Acer, Lenovo MY)
            # "coreml",    # iPhone/iPad
            # "paddle",    # China deployment
            # "saved_model" # TensorFlow web
        ]
        # self.model.overrides['task'] = 'detect'
        for fmt in export_formats:
            try:
                exported = self.model.export(
                    format=fmt,
                    imgsz=640,
                    half=True,         # FP16 = faster on mobile
                    batch=1,
                    device=0,
                    simplify=True,
                    dynamic=False,
                )
                print(f"{fmt.upper():12} → {exported}")
            except Exception as e:
                print(f"{fmt.upper():12} → FAILED ({e})")

        # Move all exported files to MY folder
        os.makedirs("tfjs_model", exist_ok=True)
        subprocess.run([
            "tensorflowjs_converter",
            "--input_format", "tf_saved_model",
            "--output_format", "tfjs_graph_model",
            "--signature_name", "serving_default",
            "--saved_model_tags", "serve",
            "saved_model",
            "tfjs_model"
        ], check=True)

    # def detect(self):
    #     self.existingModel()
    #     self.model.fuse()
    #     class_names = ["angry", "contempt", "disgust", "fear", "happy", "natural", "sad", "sleepy", "surprised"]
        
    #     try:
    #         data = request.get_json()
    #         img_b64 = data["image"].split(",")[1]
    #         img_bytes = base64.b64decode(img_b64)
    #         nparr = np.frombuffer(img_bytes, np.uint8)
    #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #         if frame is None:
    #             return jsonify({"error": "Failed to decode image"}), 400

    #         # Inference — NO .fuse(), NO extra args
    #         results = self.model(frame, imgsz=640, conf=0.5, iou=0.45, verbose=False)[0]

    #         detections = []
    #         if results.boxes is not None:
    #             boxes = results.boxes.xyxy.cpu().numpy()
    #             scores = results.boxes.conf.cpu().numpy()
    #             classes = results.boxes.cls.cpu().numpy().astype(int)

    #             for box, score, cls in zip(boxes, scores, classes):
    #                 x1, y1, x2, y2 = map(int, box)
    #                 detections.append({
    #                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    #                     "label": class_names[cls],
    #                     "conf": round(float(score), 2)
    #                 })

    #         return jsonify({"detections": detections})

    #     except Exception as e:
    #         print("Error:", e)
    #         return jsonify({"error": str(e)}), 500

# Da jadi
    # def detect(self):
    #     self.existingModel()
    #     self.model.fuse()
    #     class_names = ["angry", "contempt", "disgust", "fear", "happy", "natural", "sad", "sleepy", "surprised"]
        
    #     if 'image' not in request.files:
    #         return jsonify({"error": "No image"}), 400

    #     file = request.files['image']
    #     nparr = np.frombuffer(file.read(), np.uint8)
    #     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     if frame is None:
    #         return jsonify({"error": "Invalid image"}), 400

    #     results = self.model(frame, imgsz=640, conf=0.5, iou=0.45, verbose=False)[0]

    #     detections = []
    #     if results.boxes is not None:
    #         for box in results.boxes:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             conf = float(box.conf[0])
    #             cls = int(box.cls[0])
    #             detections.append({
    #                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    #                 "label": class_names[cls],
    #                 "conf": round(conf, 2)
    #             })
    #     return jsonify({"detections": detections})
    def detect(self):
        self.existingModel()
        self.model.fuse()
        class_names = ["angry", "contempt", "disgust", "fear", "happy", "natural", "sad", "sleepy", "surprised"]
        
        if 'image' not in request.files:
            return jsonify({"error": "No image"}), 400

        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # CONVERT TO GRAYSCALE BEFORE PREDICTION
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to 3-channel for YOLO (it expects RGB)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # YOLO inference on grayscale
        results = self.model(gray_3ch, imgsz=640, conf=0.5, iou=0.45, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": class_names[cls],
                    "conf": round(conf, 2)
                })

        return jsonify({"detections": detections})