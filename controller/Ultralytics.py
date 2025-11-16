import base64
import cv2
from flask import Blueprint, request, jsonify
import numpy as np
import service.Ultralytics as service
import torch
aiApp = service.Ultralytics()

class UltralyticsAiClass(object):

    controller = Blueprint("ultra", __name__, url_prefix="/api/ai/ultra")

    @controller.route("/initialize")
    def initializeAi():
        aiApp.a = 'initialize'
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1e9} GB")
        return 'initialize'

    @controller.route("/change/<value>")
    def changeAi(value):
        aiApp.a = value
        return aiApp.a
    
    @controller.route("/testBody", methods=['POST'])
    def bodyAi():
        data = request.json
        aiApp.a = data
        return aiApp.a
    
    @controller.route("/print")
    def valueAi():
        return aiApp.a
    
    @controller.route("/train")
    def trainAi():
        return aiApp.train()
    
    @controller.route("/performance")
    def modelPerformance():
        return aiApp.modelPerformance()
    
    @controller.route("/checkcache")
    def checkCache():
        return aiApp.checkCache()
    
    @controller.route("/export")
    def exportModel():
        return aiApp.exportModel()
    
    @controller.route("/detect", methods=["POST"])
    def detect():
        return aiApp.detect()
    