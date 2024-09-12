# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
"""
import sys
import os
from torch.utils.data import DataLoader, ConcatDataset

from Net.CDDFuse import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtraction,
    DetailFeatureExtraction,
)

from Net.DBF_KD import Restormer_Encoder as Restormer_Encoder_HI
from Net.DBF_KD import Restormer_Decoder as Restormer_Decoder_HI
from Net.DBF_KD import BaseFeatureExtraction as BaseFeatureExtraction_HI
from Net.DBF_KD import DetailFeatureExtraction as DetailFeatureExtraction_HI
from torch.utils.data import DataLoader, ConcatDataset


from utils.dataset import H5Dataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
import logging
from utils.loss import SoftLabelLoss, FeatureMapLoss


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# . Set the hyper-parameters for training
epoch1 = 4
epoch2 = 10


teacher_name = "CDDFuse_IVF"
teacher_path = f"models/{teacher_name}.pth"
std_name = f"DBF_KD"

lr = 1e-4
weight_decay = 0
batch_size = 6
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]
coeff_mse_loss_VF = 1.0  # alpha1
coeff_mse_loss_IF = 1.0
coeff_decomp = 2.0  # alpha2 and alpha4
coeff_tv = 5.0

clip_grad_norm_value = 0.01
optim_step = 2
optim_gamma = 0.5


criteria_fusion = Fusionloss()
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# TeacherModel
Encoder_t = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder_t = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer_t = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer_t = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

Encoder_t.load_state_dict(torch.load(teacher_path)["Encoder"])
Decoder_t.load_state_dict(torch.load(teacher_path)["Decoder"])
BaseFuseLayer_t.load_state_dict(torch.load(teacher_path)["BaseFuseLayer"])
DetailFuseLayer_t.load_state_dict(torch.load(teacher_path)["DetailFuseLayer"])
# StdModel
Encoder_s = nn.DataParallel(Restormer_Encoder_HI()).to(device)
Decoder_s = nn.DataParallel(Restormer_Decoder_HI()).to(device)
BaseFuseLayer_s = nn.DataParallel(BaseFeatureExtraction_HI(dim=64, num_heads=8)).to(
    device
)
DetailFuseLayer_s = nn.DataParallel(DetailFeatureExtraction_HI(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(Encoder_s.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Decoder_s.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer_s.parameters(), lr=lr, weight_decay=weight_decay
)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer_s.parameters(), lr=lr, weight_decay=weight_decay
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


visual_infrared_dataset = H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5")
medical_dataset = H5Dataset(r"data/Data4Enhance_imgsize_128_stride_200.h5")

dataLoader_viir = DataLoader(
    visual_infrared_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
dataLoader_medical = DataLoader(
    medical_dataset,
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

softlabel_diff_calculator = SoftLabelLoss(temperature=4.0).to(device)
feature_map_diff_calculator = FeatureMapLoss(reduction="mean").to(device)


Encoder_t.eval()
Decoder_t.eval()
BaseFuseLayer_t.eval()
DetailFuseLayer_t.eval()

Encoder_s.train()
Decoder_s.train()
BaseFuseLayer_s.train()
DetailFuseLayer_s.train()


for e in range(epoch1):
    dataLoader = dataLoader_viir
    print("Viir Loader")
    for i, (data_VI, data_IR) in enumerate(dataLoader):
        data_VI, data_IR = data_VI.cuda(), data_IR.cuda()
        Encoder_t.zero_grad()
        Decoder_t.zero_grad()
        BaseFuseLayer_t.zero_grad()
        DetailFuseLayer_t.zero_grad()

        Encoder_s.zero_grad()
        Decoder_s.zero_grad()
        BaseFuseLayer_s.zero_grad()
        DetailFuseLayer_s.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # Phase I
        feature_V_B, feature_V_D, _ = Encoder_t(data_VI)
        feature_I_B, feature_I_D, _ = Encoder_t(data_IR)
        feature_V_B_std, feature_V_D_std, _ = Encoder_s(data_VI)
        feature_I_B_std, feature_I_D_std, _ = Encoder_s(data_IR)

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

        data_VI_hat, data_VI_feature = Decoder_t(data_VI, feature_V_B, feature_V_D)
        data_IR_hat, data_IR_feature = Decoder_t(data_IR, feature_I_B, feature_I_D)
        data_VI_hat_std, data_VI_feature_std = Decoder_s(
            data_VI, feature_V_B_std, feature_V_D_std
        )
        data_IR_hat_std, data_IR_feature_std = Decoder_s(
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
            Encoder_s.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        nn.utils.clip_grad_norm_(
            Decoder_s.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        optimizer1.step()
        optimizer2.step()

        info = f"\r[Epoch {e}/{epoch1}],[Batch {i}/{len(dataLoader)}],[Loss:{loss.item():.5f}],[Task:{recon_loss.item():.5f}],[Soft:{softLabelLoss_encoder.item():.5f}], [Feature:{featureMapLoss_encoder.item():.5f}]"

        feature_V_B, feature_V_D, feature_V = Encoder_t(data_VI)
        feature_I_B, feature_I_D, feature_I = Encoder_t(data_IR)

        feature_F_B = BaseFuseLayer_t(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer_t(feature_I_D + feature_V_D)

        feature_F_B_std = BaseFuseLayer_s(feature_I_B + feature_V_B)
        feature_F_D_std = DetailFuseLayer_s(feature_I_D + feature_V_D)

        # SoftLabel loss
        softlabelLoss_B = softlabel_diff_calculator(feature_F_B, feature_F_B_std)
        softlabelLoss_D = softlabel_diff_calculator(feature_F_D, feature_F_D_std)
        softLabelLoss_fuseLayer = softlabelLoss_B + softlabelLoss_D

        # featureMapLoss
        featureMapLoss_B = feature_map_diff_calculator(feature_F_B, feature_F_B_std)
        featureMapLoss_D = feature_map_diff_calculator(feature_F_D, feature_F_D_std)
        featureMapLoss_fuseLayer = featureMapLoss_B + featureMapLoss_D
        data_Fuse, feature_F = Decoder_t(data_VI, feature_F_B, feature_F_D)
        data_Fuse_std, feature_F_std = Decoder_s(
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
            Encoder_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            BaseFuseLayer_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DetailFuseLayer_t.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        info = f"\r[Epoch {e}/{epoch1}],[Batch {i}/{len(dataLoader)}],[Loss:{loss.item():.5f}],[Task:{recon_loss.item():.5f}],[distill:{distillation_loss2.item():.5f}]"
        sys.stdout.write(info)


for e in range(epoch2):
    dataLoader = dataLoader_medical
    print("Medical Loader")
    for i, (data_VI, data_IR) in enumerate(dataLoader):
        data_VI, data_IR = data_VI.cuda(), data_IR.cuda()
        Encoder_t.zero_grad()
        Decoder_t.zero_grad()
        BaseFuseLayer_t.zero_grad()
        DetailFuseLayer_t.zero_grad()

        Encoder_s.zero_grad()
        Decoder_s.zero_grad()
        BaseFuseLayer_s.zero_grad()
        DetailFuseLayer_s.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        feature_V_B, feature_V_D, feature_V = Encoder_t(data_VI)
        feature_I_B, feature_I_D, feature_I = Encoder_t(data_IR)

        feature_F_B = BaseFuseLayer_t(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer_t(feature_I_D + feature_V_D)

        feature_F_B_std = BaseFuseLayer_s(feature_I_B + feature_V_B)
        feature_F_D_std = DetailFuseLayer_s(feature_I_D + feature_V_D)

        # SoftLabel loss
        softlabelLoss_B = softlabel_diff_calculator(feature_F_B, feature_F_B_std)
        softlabelLoss_D = softlabel_diff_calculator(feature_F_D, feature_F_D_std)
        softLabelLoss_fuseLayer = softlabelLoss_B + softlabelLoss_D

        # featureMapLoss
        featureMapLoss_B = feature_map_diff_calculator(feature_F_B, feature_F_B_std)
        featureMapLoss_D = feature_map_diff_calculator(feature_F_D, feature_F_D_std)
        featureMapLoss_fuseLayer = featureMapLoss_B + featureMapLoss_D
        data_Fuse, feature_F = Decoder_t(data_VI, feature_F_B, feature_F_D)
        data_Fuse_std, feature_F_std = Decoder_s(
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
            Encoder_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            BaseFuseLayer_t.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DetailFuseLayer_t.parameters(),
            max_norm=clip_grad_norm_value,
            norm_type=2,
        )
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        info = f"\r[Epoch {e}/{epoch1}],[Batch {i}/{len(dataLoader)}],[Loss:{loss.item():.5f}],[Task:{recon_loss.item():.5f}]"
        sys.stdout.write(info)


checkpoint = {
    "Encoder": Encoder_s.state_dict(),
    "Decoder": Decoder_s.state_dict(),
    "BaseFuseLayer": BaseFuseLayer_s.state_dict(),
    "DetailFuseLayer": DetailFuseLayer_s.state_dict(),
}
save_path = os.path.join(f"models/{std_name}.pth")
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
