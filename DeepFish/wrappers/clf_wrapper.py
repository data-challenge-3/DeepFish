import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from DeepFish import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from DeepFish import wrappers
from torchvision import transforms
from haven import haven_utils as hu


class ClfWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, opt: torch.optim.Optimizer, device="cuda"):
        super().__init__()
        self.model = model
        self.opt = opt
        self.device = device

    def train_on_loader(self, train_loader):
        return wrappers.train_on_loader(self, train_loader)

    def val_on_loader(self, val_loader):
        val_monitor = ClfMonitor()
        return wrappers.val_on_loader(self, val_loader, val_monitor=val_monitor)

    def vis_on_loader(self, vis_loader, savedir):
        return wrappers.vis_on_loader(self, vis_loader, savedir=savedir)

    def train_on_batch(self, batch: dict, **extras):
        self.opt.zero_grad()

        labels = batch["labels"].to(self.device)
        logits = self.model.forward(batch["images"].to(self.device))
        loss_clf =  F.binary_cross_entropy_with_logits(logits.squeeze(),
                        labels.squeeze().float(), reduction="mean")
        loss_clf.backward()

        self.opt.step()

        return {"loss_clf":loss_clf.item()}

    def val_on_batch(self, batch, **extras):
        pred_clf = self.predict_on_batch(batch)
        return (pred_clf.cpu().numpy().ravel() != batch["labels"].numpy().ravel())

    def predict_on_batch(self, batch):
        images = batch["images"].to(self.device)
        n = images.shape[0]
        logits = self.model.forward(images)
        return (torch.sigmoid(logits) > 0.5).float()

    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        # clf
        pred_labels = float(self.predict_on_batch(batch))
        img_tensor = batch["image_original"][0]  # Remove the batch dimension
        img = transforms.ToPILImage()(img_tensor)  # Convert to PIL image
        os.makedirs(f"{savedir_image}/images", exist_ok=True)
        img.save(f"{savedir_image}/images/{batch['meta']['index']}.jpg")
        hu.save_json(savedir_image+"/images/%d.json" % batch["meta"]["index"],
                    {"pred_label":float(pred_labels), "gt_label": float(batch["labels"])})


class ClfMonitor:
    def __init__(self):
        self.corrects = 0
        self.n_samples = 0

    def add(self, corrects):
        self.corrects += corrects.sum()
        self.n_samples += corrects.shape[0]

    def get_avg_score(self):
        return {"val_clf": self.corrects/ self.n_samples}
