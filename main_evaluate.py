import utils
import torch
import argparse
from models import resnet, vit
import numpy as np
from metrics import CR, MIACR

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
    parser.add_argument('--cal_sizes', type=str, default='3000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl = utils.generate_dataset(args.unlearn_type,
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

    utils.load_model(net, args.model_dir)

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    '''CR'''
    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    for cal_size in cal_size_list:
        for alpha in alpha_list:
            if args.unlearn_type == 'random':
                cov_list, size_list, acc_list, q_hat_list = CR.get_CP_data_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_size,
                                                                                batch_size=args.batch_size)
            elif args.unlearn_type == 'class':
                cov_list, size_list, acc_list, q_hat_list = CR.get_CP_class_wise(net, alpha, train_retain_dl,
                                                                                train_forget_dl, test_retain_dl,
                                                                                test_forget_dl,
                                                                                device, cal_size,
                                                                                batch_size=args.batch_size)


    '''MIACR'''
    test_len = len(test_retain_dl.dataset)
    shadow_train = torch.utils.data.Subset(test_retain_dl.dataset, list(range(test_len)))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=args.batch_size, shuffle=False
    )

    for alpha in alpha_list:
        m = MIACR.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_retain_dl,
            target_train=train_forget_dl,
            target_test=test_forget_dl,
            model=net,
            device=device,
            alpha=alpha
        )