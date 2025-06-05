Image Classification Model
==========================================

This project demonstrates how to convert a pretrained PyTorch image classification model (ResNet-18)
into ONNX format and perform inference using ONNX Runtime.


Project Structure
-----------------
model.py              - ONNX model loader, image preprocessor, and prediction logic  
convert_to_onnx.py    - Converts the PyTorch model to ONNX  
test.py               - Tests the ONNX model locally with image input  
requirements.txt      - Python dependencies  
model.onnx            - The converted ONNX model (after conversion)  
README.txt            - Project documentation  


Getting Started
---------------
1. Clone the Repository
-----------------------
git clone [https://github.com/SurbhiGurjar2003/coder_byte.git]


2. Install Dependencies
------------------------
(Optional) Create virtual environment:
python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate  

Install required packages:
pip install -r requirements.txt  


3. Convert PyTorch Model to ONNX
--------------------------------
python convert_to_onnx.py

This generates `model.onnx` from a pretrained ResNet-18 model.


4. Test ONNX Inference
----------------------
python test.py --img_path ./sample_image.jpg

This will load the ONNX model and print the predicted class index for the input image.


Model Details
-------------
- Architecture: ResNet-18 (pretrained on ImageNet)
- Format: PyTorch → ONNX
- Input Shape: 3 x 224 x 224
- Preprocessing:
    - Resize to 224x224
    - CenterCrop
    - Normalize with ImageNet stats
    - Convert to Tensor


Requirements
------------
- Python 3.8+
- PyTorch
- ONNX & ONNX Runtime
- Torchvision
- Pillow





