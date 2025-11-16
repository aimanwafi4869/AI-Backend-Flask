import wrapper.Flask as wrapper
from flask import Flask 
from controller import Api
from controller.Ultralytics import UltralyticsAiClass
import torch
import service.Ultralytics as Ultralytics
from controller.QuickDraw import QuickDrawAiClass

flaskApp = Flask(__name__)

app = wrapper.FlaskWrapper(flaskApp)

# app.addEnpoint('/api', 'api', Api, methods=['GET'])

app.registerBlueprint(Api.controller, url_prefix='/api/rest')
# app.registerBlueprint(TensorFlowAiClass.controller, url_prefix='/api/ai')
app.registerBlueprint(UltralyticsAiClass.controller, url_prefix='/api/ai/ultra')
# app.registerBlueprint(ScikitLearnAiClass.controller, url_prefix='/api/ai/sklearn')
app.registerBlueprint(QuickDrawAiClass.controller, url_prefix='/api/ai/quickdraw')

# ai = Ultralytics.Ultralytics()
# ai.train()
# ai.retrain()

# if __name__ == '__main__': 
#     app.run(debug=True)