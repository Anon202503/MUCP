import utils
import torch
import time
import os
import argparse
import unlearn_framework
import numpy as np
import wandb
from torch.utils.data import ConcatDataset
from models import resnet, vit
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    flag = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unlearn_name',
        type=str,
        default='teacher',
        choices=[
            "retrain",
            "finetune",
            "RL",
            "GA",
            "FisherForgetting",
            "teacher",
            "ssd",
        ])
    parser.add_argument(
        '--unlearn_type',
        type=str,
        default='random',
        choices=[
            "random",
            "class",
        ])
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--model_dir', type=str, default='./resnet18-cifar10/fine_model_baseline/final_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default=None)  # [82,122,163] for retrain 200 epochs None
    parser.add_argument('--seed', type=int, default=10)
    # evaluation
    parser.add_argument('--cal_size', type=int, default=2000)
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    # hyperparameters for unlearning framework
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--lamda', type=float, default=0.5)
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.unlearn_type == 'random':
        save_dir = os.path.join(args.model_name +'_' + args.data_name,
                            args.unlearn_name +'_saved_forget' + str(int(100-args.retain_ratio*100)) +'_epoch' + str(args.num_epochs))+'_loss'
    elif args.unlearn_type == 'class':
        save_dir = os.path.join(args.model_name + '_' + args.data_name,
                                args.unlearn_name + '_saved_forget_class_epoch' + str(args.num_epochs)) + str(args.num_epochs) +'_loss'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio,
                                                                                              None)
    full_train_dl = DataLoader(
        ConcatDataset((train_retain_dl.dataset, train_forget_dl.dataset)),
        batch_size=args.batch_size,
    )

    cal_train_size = args.cal_size
    cal_ds, cal_ds_2, test_ds = random_split(test_retain_dl.dataset, [cal_train_size, cal_train_size, len(test_retain_dl.dataset)-cal_train_size*2])

    cal_dl = DataLoader(dataset=cal_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_retain_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    if args.model_name == 'resnet18':
        net = resnet.ResNet18(num_classes=args.num_classes)
        unlearning_teacher = resnet.ResNet18(num_classes=args.num_classes).to(device) if args.unlearn_name == 'teacher' else None
    elif args.model_name == 'vit':
        net = vit.ViT(num_classes=args.num_classes, pretrained=False)
        unlearning_teacher = vit.ViT(num_classes=args.num_classes, pretrained=True).to(device) if args.unlearn_name == 'teacher' else None

    if args.unlearn_name != 'retrain':
        utils.load_model(net, args.model_dir)

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    # wandb
    proj_name = '{}_{}_{}_epoch{}_cp_loss'.format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs)
    watermark = "{}_lr{}_delta{}_alpha{}_lamda{}".format(args.model_name, args.learning_rate, args.delta, args.alpha, args.lamda)
    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(net)

    kwargs = {
        "model": net,
        "train_retain_dl": train_retain_dl,
        "train_forget_dl": train_forget_dl,
        "test_retain_dl": test_retain_dl,
        "test_forget_dl": test_forget_dl,
        "cal_dl": cal_dl,
        "dampening_constant": 1,   # Lambda for ssd
        "selection_weighting": 5 if args.model_name == 'vit' and args.data_name=='cifar100' else 10,   # Alpha for ssd
        "num_classes": args.num_classes,
        "dataset_name": 'cifar10',
        "device": device,
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "milestones": np.array(args.milestones.split(',')).astype(int) if args.milestones is not None else None,
        "batch_size": args.batch_size,
        "full_train_dl": full_train_dl,
        "unlearning_teacher": unlearning_teacher,
        "delta": args.delta,
        "alpha": args.alpha,
        "lamda": args.lamda,
        "unlearn_type": args.unlearn_type,
        "unlearn_name": args.unlearn_name,
    }

    start = time.time()
    acc_test = getattr(unlearn_framework, args.unlearn_name)(
        **kwargs
    )
    wandb.save("cwloss_{}_{}_{}_{}_{}.h5".format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs, args.learning_rate))
    torch.save(net.state_dict(), os.path.join(save_dir, f'final_model_delta{args.delta}_alpha{args.alpha}_lamda{args.lamda}.pth'))
