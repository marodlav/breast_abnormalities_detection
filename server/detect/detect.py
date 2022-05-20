import torch
from PIL import Image

def run(imgage_file, weights_path="weights/best.pt"):

    model = torch.hub.load("ultralytics/yolov5", 'custom', path=weights_path, force_reload=True)
    img = Image.open(imgage_file)
    results = model(img, size=640)
    results.print()
    results.show()