import argparse
from tqdm import tqdm
import os
import torch
from GCNnetwork import *
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def test(net, dataloader, loss, args):
    net.eval()
    correct = 0
    l = 0

    with torch.no_grad():
        for i, (data, adj_mat, target) in enumerate(dataloader):
            data, adj_mat, target = data.squeeze().to(args.device), adj_mat.squeeze().to(args.device), target.squeeze().to(args.device)
            output = net(data, adj_mat)

            l += loss(output, target).item()

            pred = output.argmax()  # get the index of the max log-probability
            target = target.argmax()
            correct += pred.eq(target.view_as(pred)).sum().item()

    return (100. * correct / len(dataloader.dataset), l / i)


def train(net, dataloader, dataloader_test, args):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.num_classes <= 2:
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = nn.CrossEntropyLoss()

    acc = 0
    l_test = 0
    with tqdm(range(args.epochs), unit="epoch", desc='Train') as pbar:
        for epoch in pbar:
            net.train()
            for step, (data, adj_mat, target) in enumerate(dataloader):
                data, adj_mat, target = data.squeeze().to(args.device), adj_mat.squeeze().to(args.device), target.squeeze().to(args.device)
                optimizer.zero_grad()
                output = net(data, adj_mat)
                l = loss(output, target)
                l.backward()
                optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix(loss=l.item(), accuracy=acc, l_test=l_test)

            acc, l_test = test(net, dataloader_test, loss, args)
            pbar.set_postfix(loss=l.item(), accuracy=acc, l_test=l_test)

    return net


def get_mean_std(loader):
    num_roi = 0
    mean = 0.0
    std = 0.0
    for data, _, _ in loader:
        batch_size, height, width = data.shape
        num_roi += batch_size * height * width
        mean += data.mean(axis=(0, 1, 2)).sum()
        std += data.std(axis=(0, 1, 2)).sum()

    mean /= num_roi
    std /= num_roi

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--nhid', type=list, default=[512, 256, 128])
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--pooling_ratio', type=float, default=0.5)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'


    data_path_neural = args.data_path + "/BSNIP_neural/pconn_np"
    data_path_behavior = args.data_path + "/BSNIP_behavior"

    if args.num_classes == 2:
        dataset = GCNDataset(data_path_neural, data_path_behavior, binary=True)
        args.num_classes = 1
    else:
        dataset = GCNDataset(data_path_neural, data_path_behavior)

    args.num_features = 718
    args.linear_in = 718
    for _ in range(len(args.nhid)):
        args.linear_in = int(args.pooling_ratio * args.linear_in)

    net = GCNetwork(args).to(args.device)

    #splits = random_split(dataset, [0.9, 0.1])
    splits = train_test_split(range(len(dataset.labels)), test_size=0.1, random_state=42, stratify=dataset.labels)
    #print(splits[1])
    splits[0] = Subset(dataset, splits[0])
    splits[1] = Subset(dataset, splits[1])


    mean, std = get_mean_std(DataLoader(splits[0], batch_size=1))
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    splits[0].dataset.transform = data_transforms
    splits[1].dataset.transform = data_transforms
    
    train_data = DataLoader(splits[0], batch_size=1, shuffle=True)
    test_data = DataLoader(splits[1], batch_size=1, shuffle=True)

    net = train(net, train_data, test_data, args)