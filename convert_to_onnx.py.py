import torch
from model import Classifier, BasicBlock

def convert_model_to_onnx():
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("resnet18-f37072fd.pth"))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "model.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    convert_model_to_onnx()
