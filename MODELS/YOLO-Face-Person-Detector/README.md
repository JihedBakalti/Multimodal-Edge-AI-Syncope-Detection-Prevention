---
license: agpl-3.0
pipeline_tag: object-detection
library_name: ultralytics
tags:
- yolo
- yolov8
- ultralytics
- object-detection
- computer-vision
- face-detection
- person-detection
---

# YOLOv8x Face & Person Detector

<div align="center">
  <a href="https://huggingface.co/spaces/iitolstykh/MiVOLO-Demo">
    <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in Spaces">
</a>
  <img src="images/image.png" width="500" alt="YOLO Detection Example"/>  
   <!-- <video controls autoplay loop muted playsinline width="80%">
    <source src="images/output.mp4" type="video/mp4">
  </video> -->
</div>

## Model Description

This model is a fine-tuned version of **YOLOv8x** specialized in detecting two specific classes: **Face** and **Person**. 

It has been trained on a large-scale proprietary dataset consisting of approximately 150,000 images. 
The high capacity of the YOLOv8x architecture combined with a diverse proprietary dataset ensures high accuracy and robustness in various scenarios.


## How to Use

### Installation
```bash
pip install ultralytics==8.1.0 torch==2.5.1 transformers huggingface_hub
```

### 1. Use with transformers

You can load the model using the Hugging Face transformers library by enabling custom code execution.

```python
from transformers import AutoModel
from PIL import Image
import torch

# 1. Load model with trust_remote_code=True
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "iitolstykh/YOLO-Face-Person-Detector", 
    trust_remote_code=True,
    dtype=torch_dtype,
).to(device)

# 2. Load image (You can use URL, PIL.Image or np.ndarray)
image = Image.open("path/to/your/image.jpg")
# image = cv2.imread("path/to/your/image.jpg")

# 3. Perform inference
results = model(image, conf=0.4, iou=0.7)[0]

# 4. Process results
print("Found objects:", [results.names[int(det.cls)] for det in results.boxes])
print("Boxes:", results.boxes)
# render_result(model=model.yolo, image=image, result=results).show()
```

### 2. Use with ultralytics

If you prefer the standard Ultralytics API, you can download the weights from the Hub and load them directly.

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch

# 1. Download model weights
model_path = hf_hub_download(
    repo_id="iitolstykh/YOLO-Face-Person-Detector",
    filename="yolov8x_person_face.pt",
    repo_type="model"
)

# 2. Load model
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(model_path)
model.fuse()
if torch_dtype is torch.float16:
    model.model = model.model.half()
model.to(device)

# 3. Perform inference
image = 'https://variety.com/wp-content/uploads/2023/04/MCDNOHA_SP001.jpg' 
results = model.predict(image, conf=0.4, iou=0.7, half=torch_dtype is torch.float16)

# 4. Show results
for result in results:
    boxes = result.boxes
    print("Found objects:", [result.names[int(c)] for c in boxes.cls])
```

### 3. Use with ultralyticsplus

This method automatically handles model downloading for ultralytics YOLO model.

```bash
pip install ultralyticsplus==0.1.0
```

```python
from ultralyticsplus import YOLO, render_result

# 1. Load model
model = YOLO('iitolstykh/YOLO-Face-Person-Detector')

# 2. Set model parameters
model.overrides['conf'] = 0.4
model.overrides['iou'] = 0.7
model.overrides['max_det'] = 100

# 3. Set image (You can use URL, PIL.Image or np.ndarray)
image = 'https://variety.com/wp-content/uploads/2023/04/MCDNOHA_SP001.jpg'

# 4. Perform inference
results = model.predict(image)

# 5. Show results
print("Found objects:", [results[0].names[int(det.cls)] for det in results[0].boxes])
render = render_result(model=model, image=image, result=results[0])
render.show()
```

## License

This model is based on the Ultralytics YOLOv8 architecture and inherits the **AGPL-3.0 License**.

Please refer to the official [Ultralytics Licensing](https://huggingface.co/Ultralytics/YOLOv8#license) details for more information regarding commercial usage and restrictions.

## Citation

🌟 If you find our work helpful, please consider citing our papers and leaving valuable stars

```bibtex
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}
```
```bibtex
@article{mivolo2024,
   Author = {Maksim Kuprashevich and Grigorii Alekseenko and Irina Tolstykh},
   Title = {Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation},
   Year = {2024},
   Eprint = {arXiv:2403.02302},
}
```
```bibtex
@article{cerberusdet,
   Author = {Irina Tolstykh,Michael Chernyshov,Maksim Kuprashevich},
   Title = {CerberusDet: Unified Multi-Dataset Object Detection},
   Year = {2024},
   Eprint = {arXiv:2407.12632},
}
```
