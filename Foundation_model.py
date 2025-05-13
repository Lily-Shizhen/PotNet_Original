import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from pathlib import Path
import yaml
config = yaml.safe_load(open("foundation.yaml", "r"))

# 1. Custom Dataset for processed QMOF graphs
class QMOFProcessedDataset(InMemoryDataset):
    def __init__(self, processed_dir: str, split: str, transform=None):
        super().__init__(root=processed_dir, transform=transform)
        file_map = {
            'train': 'dft_3d_train-target-infinite.pt',
            'val':   'dft_3d_val-target-infinite.pt',
            'test':  'dft_3d_test-target-infinite.pt',
        }

        path = Path(processed_dir) / file_map[split]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [
            'dft_3d_train-target-infinite.pt',
            'dft_3d_val-target-infinite.pt',
            'dft_3d_test-target-infinite.pt',
        ]

    def download(self):
        pass

    def process(self):
        pass

# 2. Masking transform applied on the fly
class MaskNodes:
    def __init__(self, mask_ratio: float = 0.2):
        self.mask_ratio = mask_ratio

    def __call__(self, data: Data) -> Data:
        num_nodes = data.x.size(0)
        num_mask = int(num_nodes * self.mask_ratio)
        perm = torch.randperm(num_nodes)
        mask_idx = perm[:num_mask]

        # store true features for the masked nodes
        data.mask_idx = mask_idx
        data.y_mask = data.x[mask_idx].clone()

        # zero out features at masked positions
        data.x[mask_idx] = 0
        return data

def main(config):
    # -- Load configuration from potnet.yaml --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_dir = config.get("process_dir", "results/processed")
    pretrain_cfg = config.get("pretrain", {})
    split        = pretrain_cfg.get("split", None)
    mask_ratio   = pretrain_cfg.get("mask_ratio", 0.2)
    batch_size   = pretrain_cfg.get("batch_size", 256)
    num_epochs   = pretrain_cfg.get("epochs", 100)
    lr           = float(pretrain_cfg.get("learning_rate", 1e-4))  # Convert to float
    # 3. Dataset and DataLoader
    if split is None:
        # concatenate train/val/test splits for self-supervised pre-training
        train_ds = ConcatDataset([
            QMOFProcessedDataset(processed_dir, s)
            for s in ('train', 'val', 'test')
        ])
    else:
        train_ds = QMOFProcessedDataset(processed_dir, split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 4. Model components
    # Adjust the import below to match your local PotNet implementation
    from models.potnet import PotNet
    encoder = PotNet(config=config.get('model', {})).to(device)
    # The PotNet encoder outputs 128-dimensional embeddings by design
    emb_dim = 128

    # Define the decoder based on embedding dimension
    node_decoder = nn.Sequential(
        nn.Linear(emb_dim, 128),
        nn.ReLU(),
        nn.Linear(128, train_ds[0].x.size(1))
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(node_decoder.parameters()),
        lr=lr
    )

    # 5. Training loop
    encoder.train()
    node_decoder.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            # -- On-the-fly masking inside the loop --
            num_nodes = batch.x.size(0)
            num_mask = int(num_nodes * mask_ratio)
            perm = torch.randperm(num_nodes, device=device)
            mask_idx = perm[:num_mask]
            y_mask = batch.x[mask_idx].clone()  # save true features
            batch.x[mask_idx] = 0  # zero-out masked nodes
            optimizer.zero_grad()
            # forward pass: get node embeddings
            node_emb = encoder(batch)  # shape [total_nodes, hidden_dim]
            # select masked nodes
            masked_emb = node_emb[mask_idx]  # index into embeddings
            # predict original features
            preds = node_decoder(masked_emb)
            # compute loss (assumes discrete atom-type labels in y_mask)
            loss  = F.cross_entropy(preds, y_mask.argmax(dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main(config)