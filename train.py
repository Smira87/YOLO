from ultralytics import YOLO
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
 
if __name__ == '__main__':

   # Load the model.
   model = YOLO('yolov8n.pt')
    
   # Training.
   results = model.train(
      data='train.yaml',
      imgsz=640,
      epochs=30,
      batch=1,
      name='yolov8_2_new1')