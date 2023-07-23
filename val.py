from ultralytics import YOLO

model_weights ="./datasets/runs/dry_run_100epochs/weights/best.pt"

# Load a model
model = YOLO(model_weights)  # load a custom model

print("model loaded")
# Validate the model
metrics = model.val(data="./configs/nano_local/data_1class.yaml",save_json=True)  # no arguments needed, dataset and settings remembered

print(f"{metrics.box.maps}")
