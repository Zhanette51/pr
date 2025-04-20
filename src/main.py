import os
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============ 0. Setup ==============
BATCH_SIZE = 128
DATA_DIR = 'data/unlabelled'  # path to unlabeled data
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 1. DataLoader ==========
# In the DATA_DIR folder there are class folders (there can be one folder if there is no markup)
transform_base = transforms.Compose([
transforms.ToTensor()
])
unlabelled_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform_base
)
unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

# ============ 2. Base Model ==========
backbone = models.resnet18(weights=None)  # Новое API
backbone.fc = nn.Identity()
projection_head = nn.Sequential(
    nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64)
)

class SimCLRModel(nn.Module):
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.head = projection_head
    def forward(self, x):
        features = self.backbone(x)
        projection = self.head(features)
        return projection

model = SimCLRModel(backbone, projection_head).to(DEVICE)

# ============ 3. Contrastive loss (NT-Xent) ==========
def nt_xent_loss(out_1, out_2, temperature=0.5):
    batch_size = out_1.shape[0]
    out_1 = nn.functional.normalize(out_1, dim=1)
    out_2 = nn.functional.normalize(out_2, dim=1)
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = ~torch.eye(2*batch_size, 2*batch_size, dtype=bool, device=out.device)
    sim_matrix = sim_matrix.masked_select(mask).view(2*batch_size, -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

# ============ 4. Augmentation ==========
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 224 for ResNet18
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor()
])

# ============ 5. Self-supervised Pre-training ==========
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(unlabelled_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    losses = []
    for (batch, _) in loop:
       # SimCLR requires two independent augmented views of each image
        x1 = torch.stack([augmentation(img) for img in batch])
        x2 = torch.stack([augmentation(img) for img in batch])
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

        out1 = model(x1)
        out2 = model(x2)
        loss = nt_xent_loss(out1, out2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loop.set_postfix({"SimCLR loss": loss.item()})
    print(f"Epoch {epoch+1}: SimCLR avg loss {sum(losses)/len(losses):.4f}")

# ======== Fine-tune (exempl) ========
# To further train the classifier:
# 1. Load the labeled data ("labelled" folder from ImageFolder)
# 2. Freeze the backbone:
#    for param in model.backbone.parameters():
#        param.requires_grad = False
# 3. Replace model.head with nn.Linear(64, NUM_CLASSES)
# 4. Train on the labeled dataset as a regular classifier.
