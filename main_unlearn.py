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
import random


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
            "scrub",
            "salun",
            "ga_plus",
            "sfron",
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
    parser.add_argument('--model_dir', type=str, default='../resnet18-cifar10/fine_model_baseline/final_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default=None)  # [82,122,163] for retrain 200 epochs None
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--mask_path', type=str, default=None, help="salun")

    # evaluation
    parser.add_argument('--cal_sizes', type=str, default='2000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()
    utils.setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.unlearn_type == 'random':
        save_dir = os.path.join(f'{args.model_name}_{args.data_name}', f'{args.unlearn_name}_new_forget{int(100 - args.retain_ratio * 100)}_epoch{args.num_epochs}')
    elif args.unlearn_type == 'class':
        save_dir = os.path.join(f'{args.model_name}_{args.data_name}', f'{args.unlearn_name}_new_forget_class_epoch{args.num_epochs}')

    os.makedirs(save_dir, exist_ok=True)

    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, cal_retain_dl, cal_forget_dl, forget_class = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio)
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
        unlearning_teacher = vit.ViT(num_classes=args.num_classes, pretrained=False).to(device) if args.unlearn_name == 'teacher' else None

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    def load_model(net, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        if list(state_dict.keys())[0].startswith("module.") and not isinstance(net, torch.nn.DataParallel):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith("module.") and isinstance(net, torch.nn.DataParallel):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

    load_model(net, args.model_dir)

    net = net.to(device)

    # wandb
    proj_name = '{}_{}_{}_epoch{}_seed'.format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs)
    watermark = "seed{}_lr{}".format(args.seed, args.learning_rate)

    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(net)


    mask = None
    if args.mask_path is not None:
        mask = torch.load(args.mask_path)


    kwargs = {
        "model": net,
        "train_retain_dl": train_retain_dl,
        "train_forget_dl": train_forget_dl,
        "test_retain_dl": test_retain_dl,
        "test_forget_dl": test_forget_dl,
        "dampening_constant": 1,   # Lambda for ssd
        "selection_weighting": 10,   # Alpha for ssd
        "num_classes": args.num_classes,
        "device": device,
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "milestones": np.array(args.milestones.split(',')).astype(int) if args.milestones is not None else None,
        "batch_size": args.batch_size,
        "full_train_dl": full_train_dl,
        "unlearning_teacher": unlearning_teacher,
        "unlearn_type": args.unlearn_type,
        "forget_class": forget_class,
        "args": args,
        "mask": mask,
        "save_dir": save_dir,
    }

    start = time.time()
    if args.unlearn_name == 'sfron':
        train_acc, forget_acc, test_acc, net = getattr(unlearn, args.unlearn_name)(
            **kwargs
        )
    else:
        train_acc, forget_acc, test_acc = getattr(unlearn, args.unlearn_name)(
            **kwargs
        )
    print('train_acc', train_acc, 'forget_acc', forget_acc, 'test_acc', test_acc)
    wandb.save("wandb_{}_{}_{}_{}_{}.h5".format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs, args.learning_rate))
    torch.save(net.state_dict(), os.path.join(save_dir, 'seed_'+str(args.seed)+'.pth'))


    print('\n\n===============================CR=============================')

    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    sheet_name = f'{args.unlearn_name}_{args.unlearn_type}'
    xlsx_path = os.path.join(save_dir, 'seed_'+str(args.seed)+'.xls')
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)


    row = 0
    for cal_size in cal_size_list:
        for alpha in alpha_list:
            if args.unlearn_type == 'random':
                cov_list, size_list, acc_list, q_hat_list = CR.get_CP_data_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_retain_dl, cal_forget_dl,
                                                                                batch_size=args.batch_size)
            elif args.unlearn_type == 'class':
                cov_list, size_list, acc_list, q_hat_list = CR.get_CP_class_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_retain_dl, cal_forget_dl,
                                                                                batch_size=args.batch_size)

            row += 1
            value = [cal_size, alpha]

            value.extend(acc_list)
            value.extend(cov_list)
            value.extend(size_list)
            # value.extend(cov_list.cpu().numpy()/size_list.cpu().numpy())
            # value.extend(np.array(cov_list)/np.array(size_list))
            value.extend(q_hat_list)

            for i, v in enumerate(value):
                sheet.write(row, i, float(v))

    print('\n\n===============================MIACR=============================')
    '''prepare data'''
    test_len = len(test_retain_dl.dataset)
    train_len = len(train_retain_dl.dataset)

    all_indices = list(range(train_len))
    shadow_indices = random.sample(all_indices, test_len)
    cal_indices = [i for i in all_indices if i not in shadow_indices]

    shadow_train_ds = torch.utils.data.Subset(train_retain_dl.dataset, shadow_indices)
    cal_ds = torch.utils.data.Subset(train_retain_dl.dataset, cal_indices)

    shadow_train_loader = torch.utils.data.DataLoader(shadow_train_ds, batch_size=args.batch_size, shuffle=False)
    cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False)

    '''MIACR'''
    # MIACR_res = MIACR.SVC_MIA(
    #     shadow_train=shadow_train_loader,
    #     shadow_test=test_retain_dl,
    #     target_train=train_forget_dl,
    #     target_test=test_forget_dl,
    #     cal_dl=cal_loader,
    #     model=net,
    #     device=device
    # )
    # MIACR_res = np.array(MIACR_res)
    # if not MIACR_res[-1]:
    #     MIACR_res = MIACR_res[:-3]
    # row += 1
    # for i, value in enumerate(MIACR_res):
    #     sheet.write(row, i, float(value))
    workbook.save(xlsx_path)