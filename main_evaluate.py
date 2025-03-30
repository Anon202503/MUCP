from torch.utils.data import Subset
import utils
import torch
import argparse
from models import resnet, vit
import numpy as np
import xlwt
import random
from metrics import CR, MIACR
import os

if __name__ == '__main__':
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
    parser.add_argument('--model_dir', type=str, default='./retraining_saved/retraining_final_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--cal_sizes', type=str, default='2000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, cal_retain_dl, cal_forget_dl, forget_class = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio)

    if args.model_name == 'resnet18':
        net = resnet.ResNet18(num_classes=args.num_classes)
    elif args.model_name == 'vit':
        net = vit.ViT(num_classes=args.num_classes, pretrained=False)


    def load_model(net, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        if list(state_dict.keys())[0].startswith("module.") and not isinstance(net, torch.nn.DataParallel):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith("module.") and isinstance(net, torch.nn.DataParallel):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)


    load_model(net, args.model_dir)

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    print('\n\n===============================CR=============================')

    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    sheet_name = f'{args.unlearn_name}_{args.unlearn_type}'
    xlsx_path = '.'+args.model_dir.split('.')[-2]
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
            value = [alpha]

            value.extend(acc_list)
            value.extend(cov_list)
            value.extend(size_list)
            # value.extend(np.array(cov_list)/np.array(size_list))
            # value.extend(cov_list.cpu().numpy()/size_list.cpu().numpy())
            value.extend(q_hat_list)

            for i, v in enumerate(value):
                sheet.write(row, i, float(v))


    print('\n\n===============================MIA=============================')
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
    MIACR_res = MIACR.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=test_retain_dl,
        target_train=train_forget_dl,
        target_test=test_forget_dl,
        cal_dl=cal_loader,
        model=net,
        device=device
    )
    MIACR_res = np.array(MIACR_res)
    if not MIACR_res[-1]:
        MIACR_res = MIACR_res[:-3]
    row += 1
    for i, value in enumerate(MIACR_res):
        sheet.write(row, i, float(value))
    workbook.save(xlsx_path)