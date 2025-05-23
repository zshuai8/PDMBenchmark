import argparse
import os


parser=argparse.ArgumentParser()
parser.add_argument('--datasets', default=['01', '02', '03', '04', '05', '06', '07', '12', '13', '16', '17', '18', '19'], nargs='+')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--models', default=["MLP", "TimeXer", "TimeMixer", "iTransformer", "PatchTST", "TimesNet", "DLinear", "Nonstationary_Transformer", "FEDformer", "Pyraformer", "Autoformer", "Informer", "Reformer", "MICN", "Crossformer", "FiLM", "SCINet", "PAttn", "FreTS"], nargs='+')
parser.add_argument('--is_training', type=int, default=0)

args=parser.parse_args()

from multiprocessing import Process

def task(dataset, model):
    # for model in args.models:
    # print("dataset: {}, model: {}".format(dataset, model))

    lr = 0.0005
    patch_len = 16
    project_input_shape = 96
    top_k = 3
    train_epochs = 100
    batch_size = args.batch_size

    if dataset in ['16']:
        patch_len = 5

    if model in ["PatchTST"] and dataset in ["01"]:
        project_input_shape = 128
    elif model in ["PatchTST"] and dataset in ["02"]:
        project_input_shape = 4864
    elif model in ["PatchTST"] and dataset in ["03"]:
        project_input_shape = 4864
    elif model in ["PatchTST"] and dataset in ["04", "05", "06", "09"]:
        project_input_shape = 128
    elif model in ["PatchTST"] and dataset in ["07"]:
        project_input_shape = 384
    elif model in ["PatchTST"] and dataset in ["12"]:
        project_input_shape = 46080
    elif model in ["PatchTST"] and dataset in ["16"]:
        project_input_shape = 15488
    elif model in ["PatchTST"] and dataset in ["17"]:
        project_input_shape = 3968
    elif model in ["PatchTST"] and dataset in ["18"]:
        project_input_shape = 4864
    elif model in ["PatchTST"] and dataset in ["19"]:
        project_input_shape = 4864
    elif model in ["PatchTST"] and dataset in ["13"]:
        project_input_shape = 4864
    elif model in ["SCINet"] and dataset in ["13"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["12"]:
        project_input_shape = 5760
    elif model in ["SCINet"] and dataset in ["09"]:
        project_input_shape = 32
    elif model in ["SCINet"] and dataset in ["04"]:
        project_input_shape = 32
    elif model in ["SCINet"] and dataset in ["02"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["03"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["01"]:
        project_input_shape = 32
    elif model in ["SCINet"] and dataset in ["05"]:
        project_input_shape = 32
    elif model in ["SCINet"] and dataset in ["06"]:
        project_input_shape = 16
    elif model in ["SCINet"] and dataset in ["16"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["17"]:
        project_input_shape = 512
    elif model in ["SCINet"] and dataset in ["18"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["19"]:
        project_input_shape = 608
    elif model in ["SCINet"] and dataset in ["07"]:
        project_input_shape = 48
    elif model in ["PAttn"] and dataset in ["02"]:
        project_input_shape = 7808
    elif model in ["PAttn"] and dataset in ["03"]:
        project_input_shape = 7808
    elif model in ["PAttn"] and dataset in ["13"]:
        project_input_shape = 7808
    elif model in ["PAttn"] and dataset in ["09"]:
        project_input_shape = 384
    elif model in ["PAttn"] and dataset in ["05"]:
        project_input_shape = 384
    elif model in ["PAttn"] and dataset in ["01", "03", "04", "06"]:
        project_input_shape = 256
    elif model in ["PAttn"] and dataset in ["07"]:
        project_input_shape = 640
    elif model in ["PAttn"] and dataset in ["12"]:
        project_input_shape = 73856
    elif model in ["PAttn"] and dataset in ["16"]:
        project_input_shape = 7808
    elif model in ["PAttn"] and dataset in ["17"]:
        project_input_shape = 6528
    elif model in ["PAttn"] and dataset in ["18"]:
        project_input_shape = 7808
    elif model in ["PAttn"] and dataset in ["19"]:
        project_input_shape = 7808
    elif model in ["TimesNet"] and dataset in ["16"]:
        top_k = 2
    elif model in ["Nonstationary_Transformer"] and dataset in ["18"]:
        lr = 0.0001
    elif model in ["Autoformer"] and dataset in ["18"]:
        lr = 0.0001
    elif model in ["Informer"] and dataset in ["18"]:
        lr = 0.0001
    elif model in ["Reformer"] and dataset in ["18"]:
        lr = 0.0001
    elif model in ["TimeXer"] and dataset in ["06"]:
        patch_len = 10

    command="nohup python -u run.py --task_name classification --is_training {} --root_path ./dataset/{}/ --model_id Heartbeat --model {} --data PDM --e_layers 3 --batch_size {} --d_model 128 --d_ff 256 --top_k {} --des Exp --itr 1 --learning_rate {} --train_epochs {} --patience {} --gpu {} --patch_len {} --project_input_shape {} > logs/log_{}_{}_{}.out".format(args.is_training, dataset, model, batch_size, top_k, str(lr), train_epochs, train_epochs, args.gpu, patch_len, project_input_shape, dataset, model, args.is_training)

    print("Running: " + command)
    os.system(command)

processes = []
for dataset in args.datasets:
    for model in args.models:
        p = Process(target=task, args=(dataset, model, ))
        processes.append(p)
        p.start()

for p in processes:
    p.join()