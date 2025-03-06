import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverper import FedPer
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from flcore.trainmodel.models import *

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):
    time_list = []
    reporter= MemReporter()
    model_str= args.model
    mimo= args.massive_mimo
    M= args.client_num
    K= args.num_terminals
    tau_p= args.pilot_num
    for i in range(args.prev, args.times):
        print(f"\n======= Running time: {i} =======")
        print("Creating server and clients")
        start= time.time()
        if model_str == "CNN":
            if mimo == "True":
                args.model = CNNModel(M, K, tau_p).to(args.device)
            else:
                if "MNIST" in args.dataset:
                    args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif model_str == "MLP":
            if mimo == "True":
                args.model = MLPModel(M, K, tau_p,n_qubits).to(args.device)
            else:
                if "MNIST" in args.dataset:
                    args.model = FedAvgMLP( num_classes=args.num_classes, dim=1024).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = FedAvgMLP( num_classes=args.num_classes, dim=1600).to(args.device)
        elif model_str == "HQCNN":
            weight_shapes = {
                    "weights_0": 3,
                    "weights_1": 3,
                    "weights_2": 1,
                    "weights_3": 1,
                    "weights_4": 1,
                    "weights_5": 3,
                    "weights_6": 3,
                }
            if mimo == "True":
                args.model = HQCNN_Ang_noQP(M, K, tau_p, n_qubits).to(args.device)
            else:
                if "MNIST" in args.dataset:
                    args.model = HQCNN_Ang_noQP(in_features=28 * 28, num_classes=args.num_classes, weight_shapes=weight_shapes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = HQCNN_Ang_noQP(in_features=3 * 32 * 32, num_classes=args.num_classes, weight_shapes=weight_shapes).to(args.device)
        else:
            raise Exception("Model not found")
        
        print(args.model)
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)  # Lấy ra đầu ra từ lớp classical
            args.model.fc = nn.Identity()  # Thay thế lớp classical output layer bằng nn.Identity()
            server = FedAvg(args, i)
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)   
        else: 
            raise Exception("Algorithm not found")
        
        # Debug: Check data consistency
        for client in server.clients:
            print(f"Client {client.id} - Number of samples: {len(client.train_data)}")
        
        server.train()
        time_list.append(time.time()-start)
    print(f"\nAvergae time cost:{ round(np.average(time_list), 2)}s.")
    
    average_data(dataset= args.dataset, algorithm= args.algorithm, goal= args.goal, times= args.times)
    print("done")
    reporter.report()
if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-kUe', "--num_terminals", type=int, default=40, help="Number of terminals (K)")
    parser.add_argument('-mAp', "--num_aps", type=int, default=100, help="Number of APs (M)")
    parser.add_argument('-mm', "--massive_mimo", type=bool, default=False,
                        help="Run in massive MIMO scenario")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)
    run(args)

