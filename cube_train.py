from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(data="/home/omp027/ComputerVison/DLR_Proj/Test_Codes/Annotation/Multiple_shapes_annot/data.yaml", epochs=100)