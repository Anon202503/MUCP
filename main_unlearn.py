from torch.utils.data import Subset
import utils
import torch
import time
import os
import argparse
import unlearn
import numpy as np
import xlwt
from metrics import CR, MIACR
import wandb
from torch.utils.data import DataLoader, ConcatDataset, dataset
from models import resnet, vit

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
    parser.add_argument('--model_dir', type=str, default='../resnet18-cifar10/original_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--forget_class', type=int, default=9)
    # evaluation
    parser.add_argument('--cal_sizes', type=str, default='3000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.unlearn_type == 'random':
        save_dir = os.path.join(args.model_name +'_' + args.data_name,
                            args.unlearn_name +'_saved_forget' + str(int(100-args.retain_ratio*100)) +'_epoch' + str(args.num_epochs))
    elif args.unlearn_type == 'class':
        save_dir = os.path.join(args.model_name + '_' + args.data_name,
                                args.unlearn_name + '_saved_forget_class_epoch' + str(args.num_epochs))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio,
                                                                                              None,
                                                                                              0)
    full_train_dl = DataLoader(
        ConcatDataset((train_retain_dl.dataset, train_forget_dl.dataset)),
        batch_size=args.batch_size,
    )

    if args.model_name == 'resnet18':
        net = resnet.ResNet18(num_classes=args.num_classes)
        unlearning_teacher = resnet.ResNet18(num_classes=args.num_classes).to(device) if args.unlearn_name == 'teacher' else None
    elif args.model_name == 'vit':
        if args.unlearn_name == 'retrain':
            net = vit.ViT(num_classes=args.num_classes, pretrained=True)
        else:
            net = vit.ViT(num_classes=args.num_classes, pretrained=False)
        unlearning_teacher = vit.ViT(num_classes=args.num_classes, pretrained=True).to(device) if args.unlearn_name == 'teacher' else None

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    if args.unlearn_name != 'retrain':
        net.load_state_dict(torch.load(args.model_dir))

    net = net.to(device)

    # wandb
    proj_name = '{}_{}_{}_epoch{}'.format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs)
    watermark = "{}_lr{}_milestones{}".format(args.model_name, args.learning_rate, args.milestones)
    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(net)

    kwargs = {
        "model": net,
        "train_retain_dl": train_retain_dl,
        "train_forget_dl": train_forget_dl,
        "test_retain_dl": test_retain_dl,
        "test_forget_dl": test_forget_dl,
        "dampening_constant": 1,   # Lambda for ssd
        "selection_weighting": 10,   # Alpha for ssd
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
        "unlearn_type": args.unlearn_type,
        "forget_class": args.forget_class,
    }

    start = time.time()
    train_acc, forget_acc, test_acc = getattr(unlearn, args.unlearn_name)(
        **kwargs
    )
    wandb.save("wandb_{}_{}_{}_{}_{}.h5".format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs, args.learning_rate))
    torch.save(net.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    for cal_size in cal_size_list:
        for alpha in alpha_list:
            print('\n\n===============================cal_size, alpha = ', cal_size, alpha, '=============================')
            if args.unlearn_type == 'random':
                cov_list, size_list, acc_list, q_hat_list = CP.get_CP_data_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_size,
                                                                                batch_size=args.batch_size)
            elif args.unlearn_type == 'class':
                cov_list, size_list, acc_list, q_hat_list = CP.get_CP_class_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_size,
                                                                                batch_size=args.batch_size)

            print('acc_list', acc_list)
            print('cov_list', cov_list)
            print('size_list', size_list)
            print('q_hat_list', q_hat_list)

    '''prepare data'''
    test_len = len(test_retain_dl.dataset)
    shadow_train = torch.utils.data.Subset(test_retain_dl.dataset, list(range(test_len)))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=args.batch_size, shuffle=False
    )


    '''SVC_MIA'''
    for alpha in alpha_list:
        m = SVC_MIA.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_retain_dl,
            target_train=train_forget_dl,
            target_test=test_forget_dl,
            model=net,
            device=device,
            alpha=alpha
        )