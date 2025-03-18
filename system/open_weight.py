import torch

# Load model
model_path = "G:\BK\Quantum\cfmimo_fed_qcnn\system\models\Cifar10\FedPer_FedAvgCNN_Cifar10_server.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))
print(type(model))
for name, param in model.state_dict().items():
    print(f"🔹 Layer: {name}, Shape: {param.shape}")
    print(param)

