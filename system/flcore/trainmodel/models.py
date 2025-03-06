import torch
import torch.nn.functional as F
from torch import nn
import pennylane as qml
import torch
import torch.nn as nn
import torch.jit
batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
### fed avg cnn
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.flatten_dim = None  # Giá trị này sẽ được tính động
        self.fc1 = nn.Identity()
        self.fc = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        if self.flatten_dim is None:
            self.flatten_dim = out.shape[1]
            self.fc1 = nn.Sequential(
                nn.Linear(self.flatten_dim, 512), 
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, self.num_classes)
            self.fc1.to(x.device)
            self.fc.to(x.device)
        
        out = self.fc1(out)
        out = self.fc(out)
        return out
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
          wires):  
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])
# Định nghĩa QNode cho HQCNN
@qml.qnode(dev)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6):
    # Amplitude Embedding
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
class HQCNN_Ang_noQP(nn.Module):
    def __init__(self, in_features, num_classes, weight_shapes):
        super(HQCNN_Ang_noQP, self).__init__()
        self.classical_layer_1 = nn.Linear(in_features, n_qubits)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc = nn.Linear(n_qubits, num_classes)
    def forward(self, x):
        # Flatten input if not already
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.classical_layer_1(x)  # Nhân input với weight của classical_layer_1
        x = self.quantum_layer(x)     # Qua quantum layer
        x = self.fc(x) 
        return x
class CNNModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = None  # To be initialized dynamically based on input size
        self.fc = nn.Linear(32, num_ue * tau_p)
        self.num_ue = num_ue
        self.tau_p = tau_p
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1, self.num_ue)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 32)
        x = self.fc1(x)
        x = self.fc(x)
        return x
class MLPModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, n_qubits):
        super(MLPModel, self).__init__()
        self.fc_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.fc_2 = nn.Linear(n_qubits, n_qubits)
        self.fc = nn.Linear(n_qubits, num_ue * tau_p)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc(x)
        return x
class mimo_HQCNN_Ang_noQP(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, weight_shapes, n_qubits):
        super(HQCNN_Ang_noQP, self).__init__()
        self.clayer_1 = nn.Linear(num_ap * num_ue, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc = nn.Linear(n_qubits, num_ue * tau_p)

    def forward(self, x):
        x = self.clayer_1(x)
        x = self.qlayer(x)
        x = self.fc(x)
        return x