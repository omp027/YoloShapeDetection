from ultralytics import YOLO

model = YOLO('/home/omp027/ComputerVison/DLR_Proj/Test_Codes/runs/detect/train3/weights/best.pt')

vid = model.predict('/home/omp027/ComputerVison/DLR_Proj/Test_Codes/1103836965-preview.mp4',show=True,conf = 0.8)

