import os
import sys
import copy
import torch
import logging
from dataloader.data_manager import DataManager

def count_parameters(model, trainable=False):
    if trainable:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6

def _set_device(args):
    device_type = args.device
    gpus = []
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args.device = gpus

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):
    from models.learner import PromptLearner as Learner
    return Learner(args)

def print_args(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        logging.info("{}: {}".format(key, value))
def train(args):
    seed_list = copy.deepcopy(args.seed)
    device = copy.deepcopy(args.device)

    init_cls = 0 if args.init_cls == args.increment else args.init_cls
    logs_name = "logs/{}/{}/Base{}-Incre{}".format(args.model_name, args.dataset, init_cls, args.increment)
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    logfilename = "logs/{}/{}/Base{}-Incre{}/{}_{}_{}".format(args.model_name, args.dataset, init_cls, args.increment, args.seed, args.backbone_type, args.suffix)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s", handlers=[logging.FileHandler(filename=logfilename + ".log"), logging.StreamHandler(sys.stdout)])

    for seed in seed_list:
        args.seed = seed
        args.device = device
        _set_random(args.seed)
        _set_device(args)
        print_args(args)
        data_manager = DataManager(args.dataset, args.shuffle, args.seed, args.init_cls, args.increment, args)
        args.nb_classes = data_manager.nb_classes
        args.nb_tasks = data_manager.nb_tasks
        model = get_model(args)
        results = {"top1": [], "top5": []}
        for task in range(data_manager.nb_tasks):
            logging.info("All params: {}M".format(count_parameters(model._network)))
            logging.info("Trainable params: {}M".format(count_parameters(model._network, True)))
            model.incremental_train(data_manager)
            eval_accy = model.eval_task()
            model.after_task()
            logging.info("CNN: {}".format(eval_accy["grouped"]))
            results["top1"].append(eval_accy["top1"])
            if (args.dataset != 'eurosat' and args.dataset != 'eurosat_in21k') and (args.dataset != 'plantvillage' and args.dataset != 'plantvillage_in21k') and (args.dataset != 'kvasir' and args.dataset != 'kvasir_in21k'):
                results["top5"].append(eval_accy["top5"])
            logging.info("CNN top1 curve: {}".format(results["top1"]))
            if (args.dataset != 'eurosat' and args.dataset != 'eurosat_in21k') and (args.dataset != 'plantvillage' and args.dataset != 'plantvillage_in21k') and (args.dataset != 'kvasir' and args.dataset != 'kvasir_in21k'):
                logging.info("CNN top5 curve: {}\n".format(results["top5"]))
            print('Average Accuracy (CNN):', sum(results["top1"]) / len(results["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(results["top1"]) / len(results["top1"])))