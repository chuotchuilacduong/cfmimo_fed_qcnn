import os
import h5py
import matplotlib.pyplot as plt

# Đường dẫn đến các file .h5
model1_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\MNIST_FedPer_CNN(32-64-4)_MLP(4-16-64-32).h5"
model2_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\MNIST_FedPer_CNN(32-64-32)_MLP(32-256-64-32).h5"
model3_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\MNIST_FedPer_Quanv_4qubit_CNN(64-16-4).h5"
model4_file = r"G:\BK\Quantum\cfmimo_fed_qcnn\results\MNIST_FedPer_Quanv_6qubit_CNN(64-16-4).h5"
dataset_name = "MNIST"

# Nhãn cho các mô hình (bạn có thể tùy chỉnh)
model_labels = [
    "CNN(32-64-4)_MLP(4-16-64-32)",
    "CNN(32-64-32)_MLP(32-256-64-32)",
    "Quanv_4qubit_CNN(64-16-4)",
    "Quanv_6qubit_CNN(64-16-4)"
]

file_paths = [model1_file, model2_file, model3_file, model4_file]

# Hàm đọc dữ liệu từ file .h5
def read_h5_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r") as h5_file:
        if "rs_test_acc" not in h5_file or "rs_train_loss" not in h5_file:
            raise KeyError(f"Missing required datasets in {file_path}. Available keys: {list(h5_file.keys())}")

        rs_test_acc = h5_file["rs_test_acc"][:]
        rs_train_loss = h5_file["rs_train_loss"][:]

    return rs_test_acc, rs_train_loss

# Đọc dữ liệu từ các file
accuracies = []
losses = []

for i, file_path in enumerate(file_paths):
    try:
        acc, loss = read_h5_data(file_path)
        accuracies.append(acc)
        losses.append(loss)
        print(f"Successfully read data from: {os.path.basename(file_path)}")
        print(f"  Model: {model_labels[i]}")
        print(f"  Number of accuracy points: {len(acc)}")
        print(f"  Number of loss points: {len(loss)}")
        if len(acc) == 0 or len(loss) == 0:
            print(f"  Warning: Empty data for accuracy or loss in {os.path.basename(file_path)}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        # Thêm dữ liệu rỗng để giữ đúng thứ tự và tránh lỗi khi vẽ
        accuracies.append([])
        losses.append([])
    except Exception as e:
        print(f"An unexpected error occurred with {os.path.basename(file_path)}: {e}")
        accuracies.append([])
        losses.append([])


# Số vòng (rounds) - giả sử tất cả các mô hình có cùng số vòng hoặc chúng ta sẽ vẽ dựa trên số vòng của từng mô hình
rounds_list = [range(1, len(acc) + 1) for acc in accuracies]

# Màu sắc và marker cho các mô hình
colors = ['blue', 'green', 'purple', 'red']
markers = ['o', 's', '^', 'D']

# Vẽ biểu đồ so sánh
plt.figure(figsize=(18, 7)) # Tăng kích thước để dễ nhìn hơn

# Biểu đồ 1: So sánh Test Accuracy
plt.subplot(1, 2, 1)
for i in range(len(accuracies)):
    if len(accuracies[i]) > 0 and len(rounds_list[i]) > 0: # Chỉ vẽ nếu có dữ liệu
        plt.plot(rounds_list[i], accuracies[i], marker=markers[i], label=f'{model_labels[i]} Acc', color=colors[i])
    elif len(accuracies[i]) == 0:
        print(f"Skipping plot for Accuracy - Model: {model_labels[i]} due to no data.")


plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title(f"Comparison of Test Accuracy on {dataset_name}")
plt.grid(True)
plt.legend(loc='best', fontsize='small') # Cải thiện vị trí và kích thước của legend

# Biểu đồ 2: So sánh Training Loss
plt.subplot(1, 2, 2)
for i in range(len(losses)):
    if len(losses[i]) > 0 and len(rounds_list[i]) > 0: # Chỉ vẽ nếu có dữ liệu
        plt.plot(rounds_list[i], losses[i], marker=markers[i], label=f'{model_labels[i]} Loss', color=colors[i]) # Sử dụng cùng màu với accuracy cho dễ theo dõi
    elif len(losses[i]) == 0:
         print(f"Skipping plot for Loss - Model: {model_labels[i]} due to no data.")

plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title(f"Comparison of Training Loss on {dataset_name}")
plt.grid(True)
plt.legend(loc='best', fontsize='small') # Cải thiện vị trí và kích thước của legend

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()