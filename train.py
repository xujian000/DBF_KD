# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import torch
import kornia
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc
from Net.DBF_KD import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtraction,
    DetailFeatureExtraction,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]

batch_size = 8
num_epochs = 10  # total epoch

criteria_fusion = Fusionloss()
result_name = f"DBF_{batch_size}_{num_epochs}"

lr = 1e-4
weight_decay = 0
coeff_mse_loss_VF = 1.0  # alpha1
coeff_mse_loss_IF = 1.0
coeff_decomp = 2.0  # alpha2 and alpha4
coeff_tv = 5.0

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay
)

scheduler1 = torch.optim.lr_scheduler.StepLR(
    optimizer1, step_size=optim_step, gamma=optim_gamma
)
scheduler2 = torch.optim.lr_scheduler.StepLR(
    optimizer2, step_size=optim_step, gamma=optim_gamma
)
scheduler3 = torch.optim.lr_scheduler.StepLR(
    optimizer3, step_size=optim_step, gamma=optim_gamma
)
scheduler4 = torch.optim.lr_scheduler.StepLR(
    optimizer4, step_size=optim_step, gamma=optim_gamma
)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(5, reduction="mean")


# data loader
trainloader = DataLoader(
    H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

loader = {
    "train": trainloader,
}
torch.backends.cudnn.benchmark = True

for epoch in range(num_epochs):
    save_path = os.path.join(f"models/{result_name}_{epoch}.pth")
    for i, (data_VI, data_IR) in enumerate(loader["train"]):
        data_VI, data_IR = data_VI.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VI)
        feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)

        data_VI_hat, _ = DIDF_Decoder(data_VI, feature_V_B, feature_V_D)
        data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)

        mse_loss_V = 5 * Loss_ssim(data_VI, data_VI_hat) + MSELoss(data_VI, data_VI_hat)
        mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

        Gradient_loss = L1Loss(
            kornia.filters.SpatialGradient()(data_VI),
            kornia.filters.SpatialGradient()(data_VI_hat),
        ) + L1Loss(
            kornia.filters.SpatialGradient()(data_IR),
            kornia.filters.SpatialGradient()(data_IR_hat),
        )

        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

        loss = (
            coeff_mse_loss_VF * mse_loss_V
            + coeff_mse_loss_IF * mse_loss_I
            + coeff_decomp * loss_decomp
            + coeff_tv * Gradient_loss
        )

        loss.backward()
        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()

        feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VI)
        feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

        data_Fuse, feature_F = DIDF_Decoder(data_VI, feature_F_B, feature_F_D)

        mse_loss_V = 5 * Loss_ssim(data_VI, data_Fuse) + MSELoss(data_VI, data_Fuse)
        mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)

        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

        fusionloss, _, _ = criteria_fusion(data_VI, data_IR, data_Fuse)

        loss = fusionloss + coeff_decomp * loss_decomp

        loss.backward()

        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Determine approximate time left
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
            % (
                epoch,
                num_epochs,
                i,
                len(loader["train"]),
                loss.item(),
            )
        )

    checkpoint = {
        "Encoder": DIDF_Encoder.state_dict(),
        "Decoder": DIDF_Decoder.state_dict(),
        "BaseFuseLayer": BaseFuseLayer.state_dict(),
        "DetailFuseLayer": DetailFuseLayer.state_dict(),
    }

    if optimizer1.param_groups[0]["lr"] <= 1e-6:
        optimizer1.param_groups[0]["lr"] = 1e-6
    if optimizer2.param_groups[0]["lr"] <= 1e-6:
        optimizer2.param_groups[0]["lr"] = 1e-6
    if optimizer3.param_groups[0]["lr"] <= 1e-6:
        optimizer3.param_groups[0]["lr"] = 1e-6
    if optimizer4.param_groups[0]["lr"] <= 1e-6:
        optimizer4.param_groups[0]["lr"] = 1e-6
    torch.save(checkpoint, save_path)
