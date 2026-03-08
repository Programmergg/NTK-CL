import argparse
from trainer import train

def get_parser():
    parser = argparse.ArgumentParser(description="Channel and Ptach Model Configuration")
    parser.add_argument("--suffix", type=str, default="", help="Prefix for logging")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the dataset")
    parser.add_argument("--model_name", type=str, default="both", choices=['channel', 'patch', 'both'], help="Model name")
    parser.add_argument("--backbone_type", type=str, default="vit_base_patch16_224_in1k", help="Backbone type")
    parser.add_argument("--device", nargs='+', default=["0"], help="Device to use")
    parser.add_argument("--seed", nargs='+', type=list, default=[0, 1, 2, 3, 4], help="Random seed")
    parser.add_argument("--init_epochs", type=int, default=20, help="Initial epochs")
    parser.add_argument("--later_epochs", type=int, default=20, help="Later epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=0.005, help="Weight decay")
    parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler")
    parser.add_argument("--ffn_num", type=int, default=16, help="Number of FFN units")
    parser.add_argument('--num_prompt_tokens', type=int, default=10, help='Number of prompt tokens')
    parser.add_argument('--hidden_size', type=int, default=768, help='Dimension of the hidden layer')
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter")
    parser.add_argument("--gamma", type=float, default=0.3, help="Gamma parameter")
    parser.add_argument("--nce_temp", type=float, default=0.2, help="NCE parameter")
    parser.add_argument("--dis_temp", type=float, default=1e-4, help="Discrimination parameter")
    parser.add_argument("--reg_temp", type=float, default=0.001, help="Regularization parameter")
    parser.add_argument("--use_init_ptm", type=bool, default=False, help="Use initial PTM")
    parser.add_argument("--use_reweight", type=bool, default=True, help="Use reweighting")
    parser.add_argument("--frozen_prompt_list_num", type=int, default=1, help="Number of Prompt Lists")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (with or without 'in21k' suffix)")
    args = parser.parse_args()
    dataset_specific_params = {
        "cifar224": {"init_cls": 10, "increment": 10, "init_lr": 0.025, "later_lr": 0.025, "weight_decay": 0.0005, "nce_temp": 0.03},
        "imagenetr": {"init_cls": 20, "increment": 20, "init_lr": 0.02, "later_lr": 0.02, "batch_size": 16},
        "imageneta": {"init_cls": 20, "increment": 20, "init_lr": 0.02, "later_lr": 0.02, "batch_size": 16},
        "oxfordpets": {"init_cls": 7, "increment": 5, "init_lr": 0.02, "later_lr": 0.02, "batch_size": 96, "use_init_ptm": True, "init_epochs": 5, "later_epochs": 5}, # 5e-4
        "eurosat": {"init_cls": 2, "increment": 2, "init_lr": 1e-4, "later_lr": 1e-4, "batch_size": 96, "use_init_ptm": True, "init_epochs": 5, "later_epochs": 5},
        "plantvillage": {"init_cls": 3, "increment": 3, "init_lr": 5e-3, "later_lr": 5e-3, "batch_size": 96, "use_init_ptm": True, "init_epochs": 5, "later_epochs": 5}, # 5e-4
        "vtab": {"init_cls": 10, "increment": 10, "init_lr": 0.03, "later_lr": 0.03, "batch_size": 96, "use_init_ptm": True, "init_epochs": 45, "later_epochs": 45},
        "kvasir": {"init_cls": 2, "increment": 2, "init_lr": 0.005, "later_lr": 0.005, "batch_size": 96, "use_init_ptm": True, "init_epochs": 5, "later_epochs": 5},
        "domainnet": {"init_cls": 23, "increment": 23, "init_lr": 0.02, "later_lr": 0.02, "batch_size": 96, "init_epochs": 50, "later_epochs": 50},
        # ImageNet-21K 预训练版本的数据集参数
        "cifar224_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "imagenetr_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "imageneta_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "oxfordpets_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "eurosat_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "kvasir_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "plantvillage_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "vtab_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
        "domainnet_in21k": {"backbone_type": "vit_base_patch16_224_in21k"},
    }
    base_dataset_name = args.dataset.replace("_in21k", "")
    if base_dataset_name in dataset_specific_params:
        for param, value in dataset_specific_params[base_dataset_name].items():
            setattr(args, param, value)
    if args.dataset in dataset_specific_params:
        for param, value in dataset_specific_params[args.dataset].items():
            setattr(args, param, value)
    return args

if __name__ == '__main__':
    args = get_parser()
    train(args)
