from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt") 

# Perform object detection on an image
results = model("/home/npnnpn_1984/hp1/hp/src/dashboard/components/assets/vid1.mp4", save=True)
for r in results:
  print(f"Detected {len(r)} objects in video")