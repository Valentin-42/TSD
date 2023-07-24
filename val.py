from ultralytics import YOLO

# Load the custom dataset
model_weights ="/mnt/gpu_storage/traffic-sign-detection/TSD/runs/detect/train/weights/best.pt"

# Load a model
model = YOLO(model_weights)  # load a custom model
print("model loaded")

# Validate the model
metrics = model.val(device=1,save_json=True, split="test")  # no arguments needed, dataset and settings remembered



print(f"{metrics.box.maps}")