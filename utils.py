import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import skimage as sk
from skimage.filters import gaussian
import torchvision.transforms.functional
from PIL import ImageFilter
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

def generate_dataset(
        unlearn_type, num_classes, batch_size, root,
        data_name, model_name, retain_ratio=0.9, shuffle=True):

    train_ds, test_ds, cal_ds = get_dataset(root, data_name, model_name)

    if unlearn_type == 'random':
        dataset_len = len(train_ds)
        retain_size = int(dataset_len * retain_ratio)
        forget_size = dataset_len - retain_size
        retain_ds, forget_ds = random_split(train_ds, [retain_size, forget_size])
        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        retain_cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return (retain_loader, forget_loader,
                test_loader, None,
                retain_cal_loader, None, -1)

    elif unlearn_type == 'class':
        random_class = random.randint(0, num_classes-1)
        print('random_class', random_class)

        forget_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] == random_class]
        retain_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] != random_class]
        forget_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] == random_class]
        retain_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] != random_class]
        forget_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] == random_class]
        retain_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] != random_class]

        forget_train_ds = Subset(train_ds, forget_indices_train)
        retain_train_ds = Subset(train_ds, retain_indices_train)
        forget_test_ds = Subset(test_ds, forget_indices_test)
        retain_test_ds = Subset(test_ds, retain_indices_test)
        forget_cal_ds = Subset(cal_ds, forget_indices_cal)
        retain_cal_ds = Subset(cal_ds, retain_indices_cal)

        train_forget_dataloader = DataLoader(forget_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        train_retain_dataloader = DataLoader(retain_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_forget_dataloader = DataLoader(forget_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_retain_dataloader = DataLoader(retain_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        cal_forget_dataloader = DataLoader(forget_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        cal_retain_dataloader = DataLoader(retain_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return (train_retain_dataloader, train_forget_dataloader,
                test_retain_dataloader, test_forget_dataloader,
                cal_retain_dataloader, cal_forget_dataloader, random_class)


def get_dataset(root, data_name, model_name):
    if data_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        image_size = 32
    elif data_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        image_size = 32
    elif data_name == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_size = 64
    else:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    if model_name == 'resnet':
        transform_train_list = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ]
    elif model_name == 'vit':
        transform_train_list = [
            transforms.Resize(224),  # Resize images to 224x224 for ViT
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.Resize(224),  # Resize images to 224x224 for ViT
            transforms.ToTensor(),
        ]
    else:
        transform_train_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

    transform_train_list.append(normalize)
    transform_test_list.append(normalize)

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    # Load dataset
    if data_name == 'cifar10':
        data_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        data_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        calibration_indices = np.random.choice(len(data_test), 2000, replace=False)
        data_cal = torch.utils.data.Subset(data_test, calibration_indices)
    elif data_name == 'cifar100':
        data_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        data_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
        calibration_indices = np.random.choice(len(data_test), 2000, replace=False)
        data_cal = torch.utils.data.Subset(data_test, calibration_indices)
    elif data_name == 'tiny_imagenet':
        data_train = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
        data_test = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test)
        data_cal = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/cal', transform=transform_test)

    return data_train, data_test, data_cal


'''
ACC metric
'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

def evaluate_acc_batch(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    return accuracy(out, clabels)

def evaluate_acc(model, val_loader, device):
    model.eval()
    corr, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            corr_, total_, _ = evaluate_acc_batch(model, batch, device)
            corr += corr_
            total += total_
    torch.cuda.empty_cache()
    return corr/total

'''unlearn'''
def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_excel_name(model_dir, corruption_type, corruption_level, unlearn_type, unlearn_name, seed):
    if unlearn_type is None:
        unlearn_type = 'None'
    # sheet_name = model_dir.split('/')[-2]
    sheet_name = f'{unlearn_name}_{unlearn_type}_{corruption_type}_{corruption_level}'
    xlsx_path = '/'.join(model_dir.split('/')[0:-1])+'/'+sheet_name+'final.xls'
    return sheet_name, xlsx_path

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(dataset_original, index):
    original_img, original_label = dataset_original[index]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # 显示原始图片
    axs[0].imshow(original_img)
    axs[0].set_title(f'Original Image\nLabel: {original_label}')

    plt.show()


