# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""
import sys
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from Net.CDDFuse import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtraction,
    DetailFeatureExtraction,
)
from Net.DBF_KD import Restormer_Encoder as Restormer_Encoder_HR
from Net.DBF_KD import Restormer_Decoder as Restormer_Decoder_HR
from Net.DBF_KD import BaseFeatureExtraction as BaseFeatureExtraction_HR
from Net.DBF_KD import DetailFeatureExtraction as DetailFeatureExtraction_HR
from torch.utils.data import DataLoader, ConcatDataset


from utils.dataset import H5Dataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
import logging
from utils.loss import DistillationLossCalculator, FeatureMapDistillationLoss


"""
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criteria_fusion = Fusionloss()

# . Set the hyper-parameters for training
epoch = 40

teacher_name = "CDDFuse_IVF"
teacher_path = f"Models/{teacher_name}.pth"
result_name = f"Enhance_{teacher_name}_epoch{epoch}_std"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(f"Logs/{result_name}.txt"), logging.StreamHandler()],
)

lr = 1e-4
weight_decay = 0
batch_size = 6
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]
# Coefficients of the loss function
coeff_mse_loss_VF = 1.0  # alpha1
coeff_mse_loss_IF = 1.0
coeff_decomp = 2.0  # alpha2 and alpha4
coeff_tv = 5.0

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

DIDF_Encoder.load_state_dict(torch.load(teacher_path)["DIDF_Encoder"])
DIDF_Decoder.load_state_dict(torch.load(teacher_path)["DIDF_Decoder"])
BaseFuseLayer.load_state_dict(torch.load(teacher_path)["BaseFuseLayer"])
DetailFuseLayer.load_state_dict(torch.load(teacher_path)["DetailFuseLayer"])

DIDF_Encoder_std = nn.DataParallel(Restormer_Encoder_HR()).to(device)
DIDF_Decoder_std = nn.DataParallel(Restormer_Decoder_HR()).to(device)
BaseFuseLayer_std = nn.DataParallel(BaseFeatureExtraction_HR(dim=64, num_heads=8)).to(
    device
)
DetailFuseLayer_std = nn.DataParallel(DetailFeatureExtraction_HR(num_layers=1)).to(
    device
)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder_std.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder_std.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer_std.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer_std.parameters(), lr=lr, weight_decay=weight_decay
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


# Loss
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")

dataset1 = H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5")
dataset2 = H5Dataset(r"data/Data4Enhance_imgsize_128_stride_200.h5")
combined_dataset = ConcatDataset([dataset1, dataset2])
trainloader = DataLoader(
    combined_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)


"""
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
"""

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

softlabel_diff_calculator = DistillationLossCalculator(temperature=4.0).to(device)
feature_map_diff_calculator = FeatureMapDistillationLoss(reduction="mean").to(device)

for e in range(epoch):
    for i, (data_VI, data_IR) in enumerate(trainloader):

        data_VI, data_IR = data_VI.cuda(), data_IR.cuda()

        DIDF_Encoder.eval()
        DIDF_Decoder.eval()
        BaseFuseLayer.eval()
        DetailFuseLayer.eval()

        DIDF_Encoder_std.train()
        DIDF_Decoder_std.train()
        BaseFuseLayer_std.train()
        DetailFuseLayer_std.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        DIDF_Encoder_std.zero_grad()
        DIDF_Decoder_std.zero_grad()
        BaseFuseLayer_std.zero_grad()
        DetailFuseLayer_std.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # Phase I
        feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VI)
        feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
        feature_V_B_std, feature_V_D_std, _ = DIDF_Encoder_std(data_VI)
        feature_I_B_std, feature_I_D_std, _ = DIDF_Encoder_std(data_IR)
        # featureMapLoss (encoder)
        featureMapLoss_VB = feature_map_diff_calculator(feature_V_B, feature_V_B_std)
        featureMapLoss_VD = feature_map_diff_calculator(feature_V_D, feature_V_D_std)
        featureMapLoss_IB = feature_map_diff_calculator(feature_I_B, feature_I_B_std)
        featureMapLoss_ID = feature_map_diff_calculator(feature_I_D, feature_I_D_std)
        # SoftLabel loss (encoder)
        vb_softlabel_loss = softlabel_diff_calculator(feature_V_B, feature_V_B_std)
        vd_softlabel_loss = softlabel_diff_calculator(feature_V_B, feature_V_B_std)
        ib_softlabel_loss = softlabel_diff_calculator(feature_V_B, feature_V_B_std)
        id_softlabel_loss = softlabel_diff_calculator(feature_V_B, feature_V_B_std)

        data_VI_hat, data_VI_feature = DIDF_Decoder(data_VI, feature_V_B, feature_V_D)
        data_IR_hat, data_IR_feature = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)
        data_VI_hat_std, data_VI_feature_std = DIDF_Decoder_std(
            data_VI, feature_V_B_std, feature_V_D_std
        )
        data_IR_hat_std, data_IR_feature_std = DIDF_Decoder_std(
            data_IR, feature_I_B_std, feature_I_D_std
        )

        # SoftLabel loss (decoder)
        softlabelLoss_VI = softlabel_diff_calculator(
            data_VI_feature, data_VI_feature_std
        )
        softlabelLoss_IR = softlabel_diff_calculator(
            data_IR_feature, data_IR_feature_std
        )
        # featureMapLoss (decoder)
        featureMapLoss_VI = feature_map_diff_calculator(
            data_VI_feature, data_VI_feature_std
        )
        featureMapLoss_IR = feature_map_diff_calculator(
            data_IR_feature, data_IR_feature_std
        )

        # recon_loss
        cc_loss_B = cc(feature_V_B_std, feature_I_B_std)
        cc_loss_D = cc(feature_V_D_std, feature_I_D_std)
        mse_loss_V = 5 * Loss_ssim(data_VI, data_VI_hat_std) + MSELoss(
            data_VI, data_VI_hat_std
        )
        mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat_std) + MSELoss(
            data_IR, data_IR_hat_std
        )
        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        Gradient_loss = L1Loss(
            kornia.filters.SpatialGradient()(data_VI),
            kornia.filters.SpatialGradient()(data_VI_hat_std),
        )

        recon_loss = (
            coeff_mse_loss_VF * mse_loss_V
            + coeff_mse_loss_IF * mse_loss_I
            + coeff_decomp * loss_decomp
            + coeff_tv * Gradient_loss
        )
        softLabelLoss_encoder = (
            vb_softlabel_loss
            + vd_softlabel_loss
            + ib_softlabel_loss
            + id_softlabel_loss
        )
        featureMapLoss_encoder = (
            featureMapLoss_VB
            + featureMapLoss_VD
            + featureMapLoss_IB
            + featureMapLoss_ID
        )
        softLabelLoss_deocder = softlabelLoss_VI + softlabelLoss_IR
        featureMapLoss_deocder = featureMapLoss_VI + featureMapLoss_IR

        distillation_loss = (
            softLabelLoss_encoder
            + featureMapLoss_encoder
            + softLabelLoss_deocder
            + featureMapLoss_deocder
        )

        loss = recon_loss + distillation_loss * 0.1  #
        loss.backward()

        nn.utils.clip_grad_norm_(
            DIDF_Encoder_std.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        nn.utils.clip_grad_norm_(
            DIDF_Decoder_std.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        optimizer1.step()
        optimizer2.step()

        info = f"\r[Epoch {e}/{epoch}],[Batch {i}/{len(trainloader)}],[Loss:{loss.item():.5f}],[Task:{recon_loss.item():.5f}],[Soft:{softLabelLoss_encoder.item():.5f}], [Feature:{featureMapLoss_encoder.item():.5f}]"

        # Phase II
        feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VI)
        feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

        feature_F_B_std = BaseFuseLayer_std(feature_I_B + feature_V_B)
        feature_F_D_std = DetailFuseLayer_std(feature_I_D + feature_V_D)

        # SoftLabel loss
        softlabelLoss_B = softlabel_diff_calculator(feature_F_B, feature_F_B_std)
        softlabelLoss_D = softlabel_diff_calculator(feature_F_D, feature_F_D_std)
        softLabelLoss_fuseLayer = softlabelLoss_B + softlabelLoss_D
        # featureMapLoss
        featureMapLoss_B = feature_map_diff_calculator(feature_F_B, feature_F_B_std)
        featureMapLoss_D = feature_map_diff_calculator(feature_F_D, feature_F_D_std)
        featureMapLoss_fuseLayer = featureMapLoss_B + featureMapLoss_D
        data_Fuse, feature_F = DIDF_Decoder(data_VI, feature_F_B, feature_F_D)
        data_Fuse_std, feature_F_std = DIDF_Decoder_std(
            data_VI, feature_F_B_std, feature_F_D_std
        )
        fusionloss, _, _ = criteria_fusion(data_VI, data_IR, data_Fuse_std)

        featureMapLoss_decoder = feature_map_diff_calculator(feature_F, feature_F_std)
        softlabelLoss_decoder = softlabel_diff_calculator(feature_F, feature_F_std)

        distillation_loss2 = (
            softLabelLoss_fuseLayer
            + featureMapLoss_fuseLayer
            + featureMapLoss_decoder
            + softlabelLoss_decoder
        )

        recon_loss = fusionloss
        loss = recon_loss + distillation_loss2 * 0.1

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

        info = f"\r[Epoch {e}/{epoch}],[Batch {i}/{len(trainloader)}],[Loss:{loss.item():.5f}],[Task:{recon_loss.item():.5f}],[Soft:{softLabelLoss_encoder.item():.5f}], [Feature:{featureMapLoss_encoder.item():.5f}]"
        sys.stdout.write(info)

    checkpoint = {
        "DIDF_Encoder": DIDF_Encoder_std.state_dict(),
        "DIDF_Decoder": DIDF_Decoder_std.state_dict(),
        "BaseFuseLayer": BaseFuseLayer_std.state_dict(),
        "DetailFuseLayer": DetailFuseLayer_std.state_dict(),
    }
    save_path = os.path.join(f"newModels/{result_name}_{e}.pth")
    torch.save(checkpoint, save_path)
    print(f"SAVE {save_path} SUCCESS")

    if optimizer1.param_groups[0]["lr"] <= 1e-6:
        optimizer1.param_groups[0]["lr"] = 1e-6
    if optimizer2.param_groups[0]["lr"] <= 1e-6:
        optimizer2.param_groups[0]["lr"] = 1e-6
    if optimizer3.param_groups[0]["lr"] <= 1e-6:
        optimizer3.param_groups[0]["lr"] = 1e-6
    if optimizer4.param_groups[0]["lr"] <= 1e-6:
        optimizer4.param_groups[0]["lr"] = 1e-6
