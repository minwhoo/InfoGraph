import time
import argparse
import datetime
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

from model import GlobalGlobalDiscriminator, GlobalPatchDiscriminator, FeedForwardNetwork

QM9_DATASET_PATH = Path("./data/QM9")


class EnnS2SEncoder(nn.Module):
    """GNN model used in Gilmer et al."""
    def __init__(self, num_features, dim):
        super().__init__()
        self.lin0 = nn.Linear(num_features, dim)

        mlp = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, mlp, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)

        out = self.set2set(out, data.batch)
        return out, feat_map[-1]


class TargetLabelSelection:
    """Set label to target label"""
    def __init__(self, target):
        self.target = target
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data


class Complete:
    """Make molecule graph complete (fully connected)"""
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class InfoGraphSemi(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim):
        super().__init__()
        self.sup_gnn_encoder = EnnS2SEncoder(input_dim, encoder_hidden_dim)
        self.unsup_gnn_encoder = EnnS2SEncoder(input_dim, encoder_hidden_dim)

        embed_dim = 2 * encoder_hidden_dim
        self.sup_unsup_discriminator = GlobalGlobalDiscriminator(
            FeedForwardNetwork(embed_dim, encoder_hidden_dim),
            FeedForwardNetwork(embed_dim, encoder_hidden_dim),
        )
        self.unsup_discriminator = GlobalPatchDiscriminator(
            FeedForwardNetwork(embed_dim, encoder_hidden_dim),
            FeedForwardNetwork(encoder_hidden_dim, encoder_hidden_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, graph_data):
        # supervised prediction
        h_sup_global, _ = self.sup_gnn_encoder(graph_data)
        pred = self.mlp(h_sup_global).view(-1)
        return pred

    def forward_unsupervised(self, graph_data):
        h_sup_global, _ = self.sup_gnn_encoder(graph_data)
        # unsupervised discriminator loss
        h_unsup_global, h_unsup_patches = self.unsup_gnn_encoder(graph_data)
        unsup_loss = self.unsup_discriminator(h_unsup_global, h_unsup_patches, graph_data.batch)

        # supervised-unsupervised discriminator loss
        sup_unsup_loss = self.sup_unsup_discriminator(h_sup_global, h_unsup_global)

        return unsup_loss, sup_unsup_loss


def train(model, labeled_dataloader, unlabeled_dataloader, optimizer, lamda, device):
    model.train()
    total_loss = 0
    for labeled_data, unlabeled_data in zip(labeled_dataloader, unlabeled_dataloader):
        labeled_data = labeled_data.to(device)
        unlabeled_data = unlabeled_data.to(device)
        optimizer.zero_grad()

        preds = model(labeled_data)
        sup_loss = F.mse_loss(preds, labeled_data.y)

        unsup_loss, sup_unsup_loss = model.forward_unsupervised(unlabeled_data)

        loss = sup_loss + unsup_loss + sup_unsup_loss * lamda
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labeled_data.num_graphs
    return total_loss / len(labeled_dataloader.dataset)


def evaluate(model, dataloader, std, device):
    model.eval()
    total_error = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            preds = model(data)
            error = F.l1_loss(preds * std, data.y * std, reduction='sum')  # MAE
            total_error += error.item()
    return total_error / len(dataloader.dataset)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    # --------------------- PARSE ARGS -----------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--target", type=int, choices=[0,1,2,3,4,5,6,7,8,9,10,11], default=0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--lamda", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)

    args = parser.parse_args()

    print("- Args ----------------------")
    for k, v in vars(args).items():
        print(" - {}={}".format(k, v))
    print("-----------------------------")

    # --------------------- LOAD DATASET ---------------------
    print("Loading dataset...")
    dataset = QM9(
        QM9_DATASET_PATH,
        pre_transform=T.Compose([Complete(), T.Distance(norm=False)]),
        transform=TargetLabelSelection(args.target)
    ).shuffle()

    mean = dataset.data.y[:,args.target].mean().item()
    std = dataset.data.y[:,args.target].std().item()
    dataset.data.y[:,args.target] = (dataset.data.y[:,args.target] - mean) / std

    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:20000+args.train_size]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    unsup_train_dataset = dataset[20000:]
    unsup_train_loader = DataLoader(unsup_train_dataset, batch_size=args.batch_size, shuffle=True)

    print("- Dataset -------------------")
    print(" - # train: {:,}".format(len(train_dataset)))
    print(" - # val: {:,}".format(len(val_dataset)))
    print(" - # test: {:,}".format(len(test_dataset)))
    print(" - # train (unsup.): {:,}".format(len(unsup_train_dataset)))
    print("-----------------------------")

    # --------------------- TRAIN MODEL ----------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraphSemi(dataset.num_features, args.encoder_hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001
    )

    val_error = evaluate(model, val_loader, std, device)
    print("| Epoch: {:3} | Val MAE: {:10.4f} |".format(0, val_error))
    print("Starting training...")

    start_time = time.time()
    checkpoint_path = "model_{}.pt".format(start_time)
    min_val_error = None
    min_val_epoch = 0
    for epoch in range(1, args.num_epoch+1):
        train_loss = train(model, train_loader, unsup_train_loader, optimizer, args.lamda, device)
        val_error = evaluate(model, val_loader, std, device)
        scheduler.step(val_error)

        if min_val_error is None or val_error < min_val_error:
            min_val_error = val_error
            min_val_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        lr = scheduler.optimizer.param_groups[0]['lr']
        elapsed_time = datetime.timedelta(seconds=int(time.time() - start_time))
        print("| Epoch: {:3} | time: {} | lr: {:7f} | Train loss: {:8.4f} | Val MAE: {:8.4f} |{}".format(
            epoch, elapsed_time, lr, train_loss, val_error, " *" if min_val_epoch == epoch else ""))

        if epoch - min_val_epoch > args.patience:
            print("Early stopping...")
            break
    print("Training finished!")

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(checkpoint_path))
    test_error = evaluate(model, test_loader, std, device)
    print("| Val MAE: {:8.4f} | Test MAE: {:8.4f} |".format(min_val_error, test_error))


if __name__ == "__main__":
    main()
