import torch
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np
import torchvision.transforms.functional
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt


def generate_dataset(unlearn_type, num_classes, batch_size, root, data_name, model_name, retain_ratio=0.9, shuffle=True):

    data_loader_train, data_loader_test, train_ds, test_ds = get_dataset(batch_size, root, data_name, model_name, shuffle)
    if unlearn_type == 'random':
        dataset_len = len(train_ds)
        retain_size = int(dataset_len * retain_ratio)
        forget_size = dataset_len - retain_size
        retain_ds, forget_ds = random_split(train_ds, [retain_size, forget_size])
        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return retain_loader, forget_loader, test_loader, None
    elif unlearn_type == 'class':
        random_class = random.randint(0, num_classes-1)

        forget_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] == random_class]
        retain_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] != random_class]
        forget_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] == random_class]
        retain_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] != random_class]

        forget_train_ds = Subset(train_ds, forget_indices_train)
        retain_train_ds = Subset(train_ds, retain_indices_train)
        forget_test_ds = Subset(test_ds, forget_indices_test)
        retain_test_ds = Subset(test_ds, retain_indices_test)

        train_forget_dataloader = DataLoader(forget_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        train_retain_dataloader = DataLoader(retain_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_forget_dataloader = DataLoader(forget_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_retain_dataloader = DataLoader(retain_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return (train_retain_dataloader, train_forget_dataloader, test_retain_dataloader, test_forget_dataloader)


def get_dataset(batch_size, root, data_name, model_name, shuffle=True):
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
    elif data_name == 'cifar100':
        data_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        data_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   pin_memory=True)
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)
    return data_loader_train, data_loader_test, data_train, data_test


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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(dataset_original, index):
    original_img, original_label = dataset_original[index]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(original_img)
    axs[0].set_title(f'Original Image\nLabel: {original_label}')

    plt.show()


def load_model(net, model_path):
    state_dict = torch.load(model_path, weights_only=True)
    if list(state_dict.keys())[0].startswith("module.") and not isinstance(net, torch.nn.DataParallel):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    elif not list(state_dict.keys())[0].startswith("module.") and isinstance(net, torch.nn.DataParallel):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)