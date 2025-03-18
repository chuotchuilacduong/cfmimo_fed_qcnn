import os
import h5py
import matplotlib.pyplot as plt

# Đường dẫn đến các file .h5
cnn_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\Cifar100_FedPer_CNN_test_0.h5"
hqcnn_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\Cifar10_FedPer_HQCNN_test_0.h5"
mlp_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\Cifar100_FedPer_MLP_test_0.h5"  # Thêm file MLP
dataset_name = "CIFAR-10"

# Hàm đọc dữ liệu từ file .h5
def read_h5_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with h5py.File(file_path, "r") as h5_file:
        if "rs_test_acc" not in h5_file or "rs_train_loss" not in h5_file:
            raise KeyError(f"Missing required datasets in {file_path}")
        
        rs_test_acc = h5_file["rs_test_acc"][:]
        rs_train_loss = h5_file["rs_train_loss"][:]
    
    return rs_test_acc, rs_train_loss

# Đọc dữ liệu từ các file
try:
    cnn_acc, cnn_loss = read_h5_data(cnn_file)
    hqcnn_acc, hqcnn_loss = read_h5_data(hqcnn_file)
    mlp_acc, mlp_loss = read_h5_data(mlp_file)  # Đọc dữ liệu MLP
except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}")
    exit()

# Số vòng (rounds)
rounds_cnn = range(1, len(cnn_acc) + 1)
rounds_hqcnn = range(1, len(hqcnn_acc) + 1)
rounds_mlp = range(1, len(mlp_acc) + 1)  # Thêm số rounds của MLP

# Vẽ biểu đồ so sánh
plt.figure(figsize=(15, 5))

# Biểu đồ 1: So sánh Test Accuracy
plt.subplot(1, 2, 1)
plt.plot(rounds_cnn, cnn_acc, marker='o', label='CNN Accuracy', color='blue')
plt.plot(rounds_hqcnn, hqcnn_acc, marker='s', label='HQCNN Accuracy', color='green')
plt.plot(rounds_mlp, mlp_acc, marker='^', label='MLP Accuracy', color='purple')  # Thêm MLP
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title(f"Comparison of Test Accuracy on {dataset_name} (HQCNN vs. CNN vs. MLP)")
plt.grid(True)
plt.legend()

# Biểu đồ 2: So sánh Training Loss
plt.subplot(1, 2, 2)
plt.plot(rounds_cnn, cnn_loss, marker='o', label='CNN Loss', color='red')
plt.plot(rounds_hqcnn, hqcnn_loss, marker='s', label='HQCNN Loss', color='orange')
plt.plot(rounds_mlp, mlp_loss, marker='^', label='MLP Loss', color='brown')  # Thêm MLP
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title(f"Comparison of Training Loss on {dataset_name} (HQCNN vs. CNN vs. MLP)")
plt.grid(True)
plt.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
