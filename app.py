from flask import Flask, request,  redirect, url_for
import cv2
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import werkzeug

app = Flask(__name__)

model = YOLO("./runs/models/best.pt")

@app.route('/')
def hello_world():
   return "Hello World"


@app.route("/detect")
def detect():
   output_result = ""
   img = cv2.imread('./images/androidFlask.jpg')
   cv2.imwrite('./images/output.jpeg', img)

   image = Image.open('./images/output.jpeg')
   detections = model.predict(image,conf=0.25,hide_labels=True,hide_conf=True,save=True)
   print("------------RESULT STARTS FROM HERE---------------")
   for r in detections:
      for c in r.boxes.cls:
        
        output_result += model.names[int(c)].lower()
        output_result += "@"
        print(model.names[int(c)])
   print("------------RESULT ENDS HERE---------------")
   return output_result

@app.route('/inputImage',methods = ['GET', 'POST'])
def inputImage():
   imagefile = request.files['image']
   filename = werkzeug.utils.secure_filename(imagefile.filename)
   print("\nReceived image File name : " + imagefile.filename)
   imagefile.save("./images/"+filename)
   return redirect(url_for('detect'))
   