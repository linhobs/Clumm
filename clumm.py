from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from ast import literal_eval
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from pprint import pprint
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
# Define the split ratios
train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    raise ModuleNotFoundError

CHECKPOINT_PATH = "/content/gdrive/MyDrive/projects/sensor placement/codes/FURI-Spring-2024/codes/simclr/ucf/center/saved_models"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()
csv_path = os.path.join(data_root, 'filtered_refined_ucf_landmarks.csv')

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)


class PoseDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        # data loading
        self.transform = transform
        xy = df.reset_index(drop=True)
        self.x = torch.tensor(xy.features).to(torch.float32)
        self.y = torch.tensor(xy.labels)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # data indexing
        sample = self.x[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[index]

    def __len__(self):
        # getting length of data
        return self.n_samples


class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
        x = x + noise
        return x


class Scaling():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(loc=1., scale=self.sigma, size=(x.shape))
        x = x * factor
        return x


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose([Jittering(0.5),
                                          Scaling(0.2),
                                          ])
train_data_ = pd.read_csv(csv_path, index_col=False,
                          converters={'features': literal_eval})
shuffled_dataset = train_data_.sample(frac=1, ignore_index=True)
train_data, temp_data = train_test_split(
    shuffled_dataset, test_size=(test_ratio + val_ratio), random_state=42)

# Split the test+validation data into test and validation
test_data, val_data = train_test_split(temp_data, test_size=(
    val_ratio / (test_ratio + val_ratio)), random_state=42)

train_dataset = PoseDataset(train_data, transform=ContrastiveTransformations(
    contrast_transforms, n_views=2))
valid_dataset = PoseDataset(
    val_data, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
test_dataset = PoseDataset(test_data)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        # self.resnet = torchvision.models.resnet18(num_classes=4*hidden_dim)  # Output of last linear layer
        self.resnet = torchvision.models.resnet18(num_classes=4*hidden_dim)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.resnet.fc = nn.Sequential(
            self.resnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        features, _ = batch
        features = torch.cat(features, dim=0).to(torch.float32)
        # Encode all images
        feats = self.resnet(features.view(-1, 1, 30, 1))
        # Calculate cosine similarity
        # explain this part very well/ maybe cosine of two different views (positive and negative)
        cos_sim = F.cosine_similarity(
            feats[:, None, :], feats[None, :, :], dim=-1)
        # print(f"cosine sim: {cos_sim}")
        # pass
        # Mask out cosine similarity to itself
        self_mask = torch.eye(
            cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        # print("NLLL",nll)
        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    # def validation_step(self, batch, batch_idx):
    #     self.info_nce_loss(batch, mode='val')


def train_simclr(batch_size, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR_Right'),
                         accelerator="gpu" if str(
                             device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(dirpath=os.path.join(CHECKPOINT_PATH, 'SimCLR_fine_tune_right'), save_weights_only=True, mode='max', monitor='train_acc_top5'),
                                    LearningRateMonitor('epoch')])
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = ""
    # pretrained_filename = "/content/gdrive/MyDrive/projects/sensor placement/codes/FURI-Spring-2024/codes/simclr/ucf/center/saved_models/SimCLR_fine_tune_right/epoch=446-step=27714.ckpt"

    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = DataLoader(dataset=train_dataset,  batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        pl.seed_everything(42)  # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        print(trainer.checkpoint_callback.best_model_path)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

    return model


simclr_model = train_simclr(batch_size=64,
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=500)
# fine-tuning


@torch.no_grad()
def prepare_data_features_(model, dataset, as_tensor_dataset=False):
    # Prepare model
    network = deepcopy(model.resnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)
    # Encode all images
    data_loader = DataLoader(
        dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.view(-1, 1, 30, 1).to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]
    if as_tensor_dataset:
        return data.TensorDataset(feats, labels)
    return np.array(feats), np.array(labels)
# img_transforms = transforms.Compose([transforms.ToTensor(),
#                                      transforms.Normalize((0.5,), (0.5,))])


train_img_data = PoseDataset(train_data)
valid_img_data = PoseDataset(val_data)
test_img_data = PoseDataset(test_data)

features, labels = prepare_data_features_(simclr_model, train_img_data)


def find_optimal_clusters(features, max_k):
    silhouette_scores = []
    # Sum of squared distances to the closest cluster center (elbow method)
    inertia = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans.inertia_)
    # Optimal K based on silhouette score
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2
    return optimal_k_silhouette, silhouette_scores, inertia


max_clusters = 10  # Define the max number of clusters to check
optimal_k_silhouette, silhouette_scores, inertia = find_optimal_clusters(
    features, max_clusters)

# Fit KMeans with the optimal number of clusters (based on silhouette)
print(
    f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")
kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=42).fit(features)
labels_ = kmeans.labels_
# shift labels so that 0 becomes 1, 1 becomes 2 etc
labels_ = labels_ + 1

# Perform PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels_,
                palette='viridis', s=100, alpha=0.8, legend='brief')
# plt.title(f'Clusters (K = {optimal_k_silhouette})
# plt.title(f'Clusters (K = {optimal_k_silhouette}) in Feature Space')
plt.xlabel('PCA Component 1', fontdict={"size": 14, "weight": "bold"})
plt.ylabel('PCA Component 2', fontdict={"size": 14, "weight": "bold"})
plt.show()

# Plot silhouette score to show optimal number of clusters based on silhouette method
plt.figure(figsize=(8, 5))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# Plot elbow method (inertia vs. number of clusters)
plt.figure(figsize=(8, 5))
plt.plot(range(2, max_clusters + 1), inertia, marker='o')
plt.title('Elbow Method: Inertia for Different Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.show()


class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.6),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')

    def predict_step(self, batch, batch_idx):
        feats, labels = batch
        preds = self.model(feats)
        return preds.argmax(dim=-1)


train_feats_simclr = prepare_data_features_(
    simclr_model, valid_img_data, as_tensor_dataset=True)
test_feats_simclr = prepare_data_features_(
    simclr_model, test_img_data, as_tensor_dataset=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}
predicted_targets = []
actual_targets = []
combined_dataset = ConcatDataset([train_feats_simclr, test_feats_simclr])


def train_logreg_kfold(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=200, **kwargs):

    for fold, (train_idx, val_idx) in enumerate(kf.split(combined_dataset)):
        train_set = Subset(combined_dataset, train_idx)
        val_set = Subset(combined_dataset, val_idx)

    # Data loaders
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                       drop_last=False, pin_memory=True, num_workers=0)
        test_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                      drop_last=False, pin_memory=True, num_workers=0)
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                             accelerator="gpu" if str(
                                 device).startswith("cuda") else "cpu",
                             devices=1,
                             max_epochs=max_epochs,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                        LearningRateMonitor("epoch")],
                             enable_progress_bar=False,
                             check_val_every_n_epoch=10)
        trainer.logger._default_hp_metric = None
    # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(
            CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt_")
        if os.path.isfile(pretrained_filename):
            print(
                f"Found pretrained model at {pretrained_filename}, loading...")
            model = LogisticRegression.load_from_checkpoint(
                pretrained_filename)
        else:
            print("Just verifying if this is r")
            pl.seed_everything(42)  # To be reproducable
            model = LogisticRegression(**kwargs)
            trainer.fit(model, train_loader, test_loader)
            model = LogisticRegression.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path)

        # Test best model on train and validation set
        train_result = trainer.test(model, train_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        predictions = trainer.predict(model, test_loader)
        predicted_targets.extend(predictions)
        for _, target in test_loader:
            actual_targets.append(target)
        results[fold] = {"train": train_result[0]
                         ["test_acc"], "test": test_result[0]["test_acc"]}

    return model, results


_, small_set_results = train_logreg_kfold(batch_size=32,
                                          train_feats_data=train_feats_simclr,
                                          test_feats_data=test_feats_simclr,
                                          model_suffix=200,
                                          feature_dim=train_feats_simclr.tensors[0].shape[1],
                                          num_classes=8,
                                          lr=1e-3,
                                          weight_decay=1e-3)
results = small_set_results
pprint(results)
test_accuracy = np.mean([r['test'] for r in results.values()])
train_accuracy = np.mean([r['train'] for r in results.values()])
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Train accuracy: {train_accuracy:.4f}")

# prompt: plot a confusion matrix using the actual target and predicted targets
actual_targets_ = torch.cat(actual_targets)
predicted_targets_ = torch.cat(predicted_targets)
print(actual_targets)
# Create the confusion matrix
cm = confusion_matrix(actual_targets_, predicted_targets_, labels=[0, 1, 2])

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("confusion matrix for fine-tuned CLUMM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# {'bend': 0, 'idle': 1, 'mid_lift': 2}
precision = precision_score(
    actual_targets_, predicted_targets_, average='macro')
recall = recall_score(actual_targets_, predicted_targets_, average='macro')
f1 = f1_score(actual_targets_, predicted_targets_, average='macro')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
