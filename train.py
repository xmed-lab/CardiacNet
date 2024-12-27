# CUDA_VISIBLE_DEVICES=1,2,4,5 
import os
import argparse
from tqdm import tqdm
import numpy as np
from einops import rearrange

import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from model.time_vqgan_template import Classifier, VQGAN, NLayerDiscriminator, NLayerDiscriminator3D
from model.sinkhorn import SinkhornSolver, sinkhorn_rpm, sinkhorn, sinkhorn_cross_batch

from utils.tools import get_world_size, get_global_rank, get_local_rank, get_master_ip
from utils.lpips import LPIPS
from utils.utils import load_data, weights_init, adopt_weight

from data.cardiacnet import CardiacNet_Dataset
from monai.data import DataLoader

import wandb

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class CardiacNet:
    def __init__(self, args):
        self.classifier = Classifier(args).to(device=args.device)
        self.opt_classifier = torch.optim.Adam(self.classifier.parameters(), lr=args.learning_rate, eps=1e-08)

        self.normal_vqgan = VQGAN(args).to(device=args.device)
        # Currently we train from scratch
        # normal_checkpoint_path = f"./checkpoints/xxxx.pt"
        # self.normal_vqgan.load_state_dict({k.replace('module.',''):v for k, v in torch.load(normal_checkpoint_path, map_location=args.device).items()})
        self.normal_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers).to(device=args.device)
        self.normal_discriminator_3d = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers).to(device=args.device)
        self.normal_discriminator.apply(weights_init)

        self.opt_vq_normal, self.opt_disc_normal, self.opt_3d_disc_normal = self.configure_optimizers(args, self.normal_vqgan, self.normal_discriminator, self.normal_discriminator_3d)

        self.abnorm_vqgan = VQGAN(args).to(device=args.device)
        # Currently we train from scratch
        # abnorm_checkpoint_path = f"./checkpoints/xxxx.pt"
        # self.abnorm_vqgan.load_state_dict({k.replace('module.',''):v for k, v in torch.load(abnorm_checkpoint_path, map_location=args.device).items()})
        self.abnorm_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers).to(device=args.device)
        self.abnorm_discriminator_3d = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers).to(device=args.device)
        self.abnorm_discriminator.apply(weights_init)
            
        self.opt_vq_abnorm, self.opt_disc_abnorm, self.opt_3d_disc_abnorm = self.configure_optimizers(args, self.abnorm_vqgan, self.abnorm_discriminator, self.abnorm_discriminator_3d)

        if args.distributed:
            self.classifier = torch.nn.parallel.DistributedDataParallel(self.classifier, broadcast_buffers=True, find_unused_parameters=True,)

            # For normal
            self.normal_vqgan = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.normal_vqgan)
            self.normal_vqgan = torch.nn.parallel.DistributedDataParallel(self.normal_vqgan, broadcast_buffers=True, find_unused_parameters=True,)
            
            self.normal_discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.normal_discriminator)
            self.normal_discriminator = torch.nn.parallel.DistributedDataParallel(self.normal_discriminator, broadcast_buffers=True, find_unused_parameters=True,)

            self.normal_discriminator_3d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.normal_discriminator_3d)
            self.normal_discriminator_3d = torch.nn.parallel.DistributedDataParallel(self.normal_discriminator_3d, broadcast_buffers=True, find_unused_parameters=True,)

            # For abnormal
            self.abnorm_vqgan = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.abnorm_vqgan)
            self.abnorm_vqgan = torch.nn.parallel.DistributedDataParallel(self.abnorm_vqgan, broadcast_buffers=True, find_unused_parameters=True,)

            self.abnorm_discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.abnorm_discriminator)
            self.abnorm_discriminator = torch.nn.parallel.DistributedDataParallel(self.abnorm_discriminator, broadcast_buffers=True, find_unused_parameters=True,)

            self.abnorm_discriminator_3d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.abnorm_discriminator_3d)
            self.abnorm_discriminator_3d = torch.nn.parallel.DistributedDataParallel(self.abnorm_discriminator_3d, broadcast_buffers=True, find_unused_parameters=True,)
        
        elif len(args.enable_GPUs_id) > 1:
            # For normal
            self.normal_vqgan = torch.nn.DataParallel(self.normal_vqgan, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
            self.normal_discriminator = torch.nn.DataParallel(self.normal_discriminator, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
            self.normal_discriminator_3d = torch.nn.DataParallel(self.normal_discriminator_3d, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])

            # For abnormal
            self.abnorm_vqgan = torch.nn.DataParallel(self.abnorm_vqgan, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
            self.abnorm_discriminator = torch.nn.DataParallel(self.abnorm_discriminator, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
            self.abnorm_discriminator_3d = torch.nn.DataParallel(self.abnorm_discriminator_3d, device_ids=args.enable_GPUs_id, output_device=args.enable_GPUs_id[0])
        
        self.mseloss = torch.nn.MSELoss(reduction='mean')
        self.cycle_criterion = torch.nn.L1Loss()
        self.cossim = torch.nn.CosineSimilarity(dim=1)
        self.sinkhorn_loss = SinkhornSolver(epsilon=1e-6, iterations=10, reduction='mean')

        # use the perceputal loss
        # self.perceptual_loss = LPIPS().eval().to(device=args.device)
        
        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        
        self.prepare_training()

        train_dataset_normal = CardiacNet_Dataset(args, select_set=['Non-ASD'], is_video=True, is_train=True)
        train_dataset_abnorm = CardiacNet_Dataset(args, select_set=['ASD'], is_video=True, is_train=True)
        train_loader_normal = DataLoader(train_dataset_normal, batch_size=args.batch_size, shuffle=True, num_workers=8)

        self.train(args, train_loader_normal, train_dataset_abnorm)

    def configure_optimizers(self, args, vq_model, vq_discriminator, vq_discriminator_3d):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(vq_model.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(vq_discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        
        opt_3d_disc = torch.optim.Adam(vq_discriminator_3d.parameters(),
                                       lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc, opt_3d_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args, train_normal, train_dataset_abnorm):
        steps_per_epoch = len(train_normal)
        record_steps = 0
        for epoch in range(args.epochs):
            train_abnorm = iter(DataLoader(train_dataset_abnorm, batch_size=args.batch_size, shuffle=True, num_workers=8))
            with tqdm(range(steps_per_epoch)) as pbar:
                for step, (normal_vids, normal_masked_pos, normal_mpap, _) in zip(pbar, train_normal):
                    abnorm_vids, abnorm_masked_pos, abnorm_mpap, _ = next(train_abnorm)

                    normal_vids = normal_vids.to(device=args.device)
                    abnorm_vids = abnorm_vids.to(device=args.device)
                    
                    normal_masked_pos = rearrange(normal_masked_pos.to(device=args.device), 'b (l h w) p1 p2 -> b l (h p1) (w p2)', 
                                                    h=args.image_size[0]//args.mask_size, w=args.image_size[1]//args.mask_size, 
                                                    l=args.image_size[2], p1=args.mask_size, p2=args.mask_size).unsqueeze(1)
                    abnorm_masked_pos = rearrange(abnorm_masked_pos.to(device=args.device), 'b (l h w) p1 p2 -> b l (h p1) (w p2)', 
                                                    h=args.image_size[0]//args.mask_size, w=args.image_size[1]//args.mask_size, 
                                                    l=args.image_size[2], p1=args.mask_size, p2=args.mask_size).unsqueeze(1)

                    video_recon_A, frame_A, frame_recon_A, (ft_N2A, vq_N2A, inter_ft_N2A_loss, inter_vq_N2A_loss, pos_emb_N2A, normal_commitment_loss), ft_output_A = self.normal_vqgan(torch.cat((normal_vids, abnorm_vids), dim=0), 
                                                                                                   torch.cat((normal_masked_pos, abnorm_masked_pos), dim=0), templated = True, type = 'N2A')
                    video_recon_B, frame_B, frame_recon_B, (ft_A2N, vq_A2N, inter_ft_A2N_loss, inter_vq_A2N_loss, pos_emb_A2N, abnorm_commitment_loss), ft_output_B = self.abnorm_vqgan(torch.cat((normal_vids, abnorm_vids), dim=0), 
                                                                                                   torch.cat((normal_masked_pos, abnorm_masked_pos), dim=0), templated = True, type = 'A2N')

                    normal2abnorm_video_recon, identity_video_A = video_recon_A[:args.batch_size], video_recon_A[args.batch_size:]
                    abnorm2normal_video_recon, identity_video_B = video_recon_B[args.batch_size:], video_recon_B[:args.batch_size]

                    normal2abnorm_frame_recon, identity_frame_A = frame_recon_A[:args.batch_size], frame_recon_A[args.batch_size:]
                    abnorm2normal_frame_recon, identity_frame_B = frame_recon_B[args.batch_size:], frame_recon_B[:args.batch_size]

                    normal_frames, abnorm_frames = frame_A[:args.batch_size], frame_A[args.batch_size:]

                    recovered_normal_video, _, recovered_normal_frame, (_, _, recovered_abnorm_commitment_loss), recovered_normal_ft_output = self.abnorm_vqgan(normal2abnorm_video_recon, 
                                                                                                                                           abnorm_masked_pos, type = 'A2N')
                    recovered_abnorm_video, _, recovered_abnorm_frame, (_, _, recovered_normal_commitment_loss), recovered_abnorm_ft_output = self.normal_vqgan(abnorm2normal_video_recon, 
                                                                                                                                           normal_masked_pos, type = 'N2A')
                    disc_factor = adopt_weight(record_steps, threshold=args.disc_start)
                    record_steps += 1

                    feat_weights = 4.0 / (3 + 1)
                    if record_steps > args.disc_start:

                        loss_identity_A = torch.sum(torch.mul(args.identity_weight, F.l1_loss(identity_video_A, abnorm_vids)))
                        loss_identity_B = torch.sum(torch.mul(args.identity_weight, F.l1_loss(identity_video_B, normal_vids)))

                        # FT_deform_loss = 0.001 * (sinkhorn(ft_N2A, ft_A2N.detach(), w_x=pos_emb_N2A, w_y=pos_emb_N2A)[0] + 
                        #                           sinkhorn(ft_A2N, ft_N2A.detach(), w_x=pos_emb_A2N, w_y=pos_emb_A2N)[0]) / 2
                        VQ_deform_loss = 0.001 * (sinkhorn_cross_batch(vq_N2A, vq_A2N.detach())[0] + 
                                                  sinkhorn_cross_batch(vq_A2N, vq_N2A.detach())[0]) / 2

                        # Discriminator for Normal Cases
                        gan_normal_feat_loss = 0
                        
                        logits_image_normal_fake, pred_image_normal_fake = self.normal_discriminator(recovered_normal_frame)
                        logits_video_normal_fake, pred_video_normal_fake = self.normal_discriminator_3d(recovered_normal_video)

                        if args.gan_feat_loss:
                            image_normal_gan_feat_loss = 0
                            video_normal_gan_feat_loss = 0
                            if args.image_gan_weight > 0:
                                logits_normal_image_real, pred_normal_image_real = self.normal_discriminator(normal_frames)
                                for i in range(len(pred_image_normal_fake)-1):
                                    image_normal_gan_feat_loss += feat_weights * F.l1_loss(pred_image_normal_fake[i], pred_normal_image_real[i].detach()) * (args.image_gan_weight > 0)
                            if args.video_gan_weight > 0:
                                logits_normal_video_real, pred_normal_video_real = self.normal_discriminator_3d(normal_vids)
                                for i in range(len(pred_video_normal_fake)-1):
                                    video_normal_gan_feat_loss += feat_weights * F.l1_loss(pred_video_normal_fake[i], pred_normal_video_real[i].detach()) * (args.video_gan_weight > 0)

                            gan_normal_feat_loss = disc_factor * args.gan_feat_weight * (image_normal_gan_feat_loss + video_normal_gan_feat_loss)
                        
                        g_normal_image_loss = -torch.mean(logits_image_normal_fake)
                        g_normal_video_loss = -torch.mean(logits_video_normal_fake)

                        logits_image_abnorm2normal_fake, _ = self.normal_discriminator(abnorm2normal_frame_recon)
                        logits_video_abnorm2normal_fake, _ = self.normal_discriminator_3d(abnorm2normal_video_recon)

                        g_normal_image_loss = g_normal_image_loss - torch.mean(logits_image_abnorm2normal_fake)
                        g_normal_video_loss = g_normal_video_loss - torch.mean(logits_video_abnorm2normal_fake)
                        g_normal_loss = args.image_gan_weight * g_normal_image_loss + args.video_gan_weight * g_normal_video_loss

                        normal_aeloss = 1 * disc_factor * g_normal_loss 
                        normal_recall_loss = 0.01 * self.mseloss(self.classifier(ft_output_A).squeeze(-1), torch.cat((normal_mpap, abnorm_mpap)).to(device=args.device).float())
                        
                        overall_normal_loss = normal_aeloss + gan_normal_feat_loss + 0.5 * loss_identity_A * 10 +\
                                              normal_commitment_loss + recovered_normal_commitment_loss + inter_ft_N2A_loss + inter_vq_N2A_loss + normal_recall_loss

                        # discriminator for abnormal cases
                        gan_abnorm_feat_loss = 0
                        logits_image_abnorm_fake, pred_image_abnorm_fake = self.abnorm_discriminator(recovered_abnorm_frame)
                        logits_video_abnorm_fake, pred_video_abnorm_fake = self.abnorm_discriminator_3d(recovered_abnorm_video)

                        if args.gan_feat_loss:
                            image_abnorm_gan_feat_loss = 0
                            video_abnorm_gan_feat_loss = 0
                            if args.image_gan_weight > 0:
                                logits_abnorm_image_real, pred_abnorm_image_real = self.abnorm_discriminator(abnorm_frames)
                                for i in range(len(pred_image_abnorm_fake)-1):
                                    image_abnorm_gan_feat_loss += feat_weights * F.l1_loss(pred_image_abnorm_fake[i], pred_abnorm_image_real[i].detach()) * (args.image_gan_weight > 0)
                            if args.video_gan_weight > 0:
                                logits_abnorm_video_real, pred_abnorm_video_real = self.abnorm_discriminator_3d(abnorm_vids)
                                for i in range(len(pred_video_abnorm_fake)-1):
                                    video_abnorm_gan_feat_loss += feat_weights * F.l1_loss(pred_video_abnorm_fake[i], pred_abnorm_video_real[i].detach()) * (args.video_gan_weight > 0)
                            gan_abnorm_feat_loss = disc_factor * args.gan_feat_weight * (image_abnorm_gan_feat_loss + video_abnorm_gan_feat_loss)
                        
                        g_abnorm_image_loss = -torch.mean(logits_image_abnorm_fake)
                        g_abnorm_video_loss = -torch.mean(logits_video_abnorm_fake)
                        
                        logits_image_normal2abnorm_fake, _ = self.abnorm_discriminator(normal2abnorm_frame_recon)
                        logits_video_normal2abnorm_fake, _ = self.abnorm_discriminator_3d(normal2abnorm_video_recon)

                        g_abnorm_image_loss = g_abnorm_image_loss - torch.mean(logits_image_normal2abnorm_fake)
                        g_abnorm_video_loss = g_abnorm_video_loss - torch.mean(logits_video_normal2abnorm_fake)

                        g_abnorm_loss = args.image_gan_weight * g_abnorm_image_loss + args.video_gan_weight * g_abnorm_video_loss

                        abnorm_aeloss = 1 * disc_factor * g_abnorm_loss
                        abnorm_recall_loss = 0.01 * self.mseloss(self.classifier(ft_output_B).squeeze(-1), torch.cat((normal_mpap, abnorm_mpap)).to(device=args.device).float())

                        overall_abnorm_loss = abnorm_aeloss + gan_abnorm_feat_loss + 0.5 * loss_identity_B * 10 +\
                                              abnorm_commitment_loss + recovered_abnorm_commitment_loss + inter_ft_A2N_loss + inter_vq_A2N_loss + abnorm_recall_loss

                        self.opt_classifier.zero_grad()
                        self.opt_vq_normal.zero_grad()
                        self.opt_vq_abnorm.zero_grad()

                        loss_cycle_A = torch.sum(torch.mul(args.cycle_weight, self.cycle_criterion(recovered_normal_video, normal_vids)))
                        loss_cycle_B = torch.sum(torch.mul(args.cycle_weight, self.cycle_criterion(recovered_abnorm_video, abnorm_vids)))
                        overall_normal_loss = overall_normal_loss + 10 * loss_cycle_A
                        overall_abnorm_loss = overall_abnorm_loss + 10 * loss_cycle_B
                        
                        overall_loss = overall_normal_loss + overall_abnorm_loss - VQ_deform_loss
                        
                        overall_loss.backward()
                        self.opt_classifier.step()
                        self.opt_vq_normal.step()
                        self.opt_vq_abnorm.step()

                        if args.local_rank == args.enable_GPUs_id[0]:
                            if args.train_normal:
                                if args.wandb:
                                    wandb.log({'Normal-loss/Recon Loss': loss_identity_A.item(), 
                                            'Normal-loss/Cycle Loss': loss_cycle_A.item(), 
                                            #'Normal-loss/Inter N2A OT Loss': inter_ft_N2A_loss.item(),
                                            'Normal-loss/Inter N2A VQ Loss': inter_vq_N2A_loss.item(), 
                                            # 'Normal-loss/FT N2A Loss': FT_deform_loss.item(), 
                                            'Normal-loss/VQ N2A Loss': VQ_deform_loss.item(), 
                                            'Normal-loss/Commitment Loss': normal_commitment_loss.item() + recovered_normal_commitment_loss.item(), 
                                            'Normal-loss/AEloss': normal_aeloss.item(),
                                            # 'Normal-loss/GAN Feature Loss': gan_normal_feat_loss.item(),
                                            'Normal-loss/Class Loss': normal_recall_loss, 
                                            'Normal-loss/Overall Loss': overall_normal_loss.item()},
                                            step = step)
                            if args.train_abnorm:
                                if args.wandb:
                                    wandb.log({'Abnorm-loss/Recon Loss': loss_identity_B.item(), 
                                            'Abnorm-loss/Cycle Loss': loss_cycle_B.item(), 
                                            # 'Abnorm-loss/Inter A2N OT Loss': inter_ft_A2N_loss.item(),
                                            'Abnorm-loss/Inter A2N VQ Loss': inter_vq_A2N_loss.item(), 
                                            # 'Abnorm-loss/FT A2N Loss': FT_deform_loss.item(), 
                                            'Abnorm-loss/VQ A2N Loss': VQ_deform_loss.item(), 
                                            'Abnorm-loss/Commitment Loss': abnorm_commitment_loss.item() + recovered_abnorm_commitment_loss.item(), 
                                            'Abnorm-loss/AEloss': abnorm_aeloss.item(),
                                            # 'Abnorm-loss/GAN Feature Loss': gan_abnorm_feat_loss.item(),
                                            'Abnorm-loss/Class Loss': abnorm_recall_loss,
                                            'Abnorm-loss/Overall Loss': overall_abnorm_loss.item()},
                                            step = step)

                    if step % 2 == 1 and record_steps > args.disc_start:
                        if args.train_normal:
                            d_normal_image_loss, d_normal_video_loss, normal_discloss = 0, 0, 0

                            logits_normal_image_real, _ = self.normal_discriminator(normal_frames.detach())
                            logits_normal_video_real, _ = self.normal_discriminator_3d(normal_vids.detach())
                            
                            logits_normal_image_fake, _ = self.normal_discriminator(recovered_normal_frame.detach())
                            logits_normal_video_fake, _ = self.normal_discriminator_3d(recovered_normal_video.detach())

                            logits_abnorm2normal_image_fake, _ = self.normal_discriminator(abnorm2normal_frame_recon.detach())
                            logits_abnorm2normal_video_fake, _ = self.normal_discriminator_3d(abnorm2normal_video_recon.detach())

                            d_normal_image_loss = self.disc_loss(logits_normal_image_real, logits_normal_image_fake) + \
                                                  self.disc_loss(logits_normal_image_real, logits_abnorm2normal_image_fake)
                                                  # self.calculate_gradient_penalty(self.normal_discriminator, normal_frames.data, recovered_normal_frame.data, args.device) * 10
                                                  # self.calculate_gradient_penalty(self.normal_discriminator, normal_frames.data, abnorm2normal_frame_recon.data, args.device) * 10
                            d_normal_video_loss = self.disc_loss(logits_normal_video_real, logits_normal_video_fake) + \
                                                  self.disc_loss(logits_normal_video_real, logits_abnorm2normal_video_fake)
                                                  # self.calculate_gradient_penalty(self.normal_discriminator_3d, normal_vids.data, recovered_normal_video.data, args.device) * 10
                                                  # self.calculate_gradient_penalty(self.normal_discriminator_3d, normal_vids.data, abnorm2normal_video_recon.data, args.device) * 10

                            normal_discloss = disc_factor * (args.image_gan_weight * d_normal_image_loss + args.video_gan_weight * d_normal_video_loss)

                            self.opt_disc_normal.zero_grad()
                            self.opt_3d_disc_normal.zero_grad()
                            normal_discloss.backward()
                            self.opt_disc_normal.step()
                            self.opt_3d_disc_normal.step()

                        if args.train_abnorm:
                            logits_abnorm_image_real, _ = self.abnorm_discriminator(abnorm_frames.detach())
                            logits_abnorm_video_real, _ = self.abnorm_discriminator_3d(abnorm_vids.detach())

                            logits_abnorm_image_fake, _ = self.abnorm_discriminator(recovered_abnorm_frame.detach())
                            logits_abnorm_video_fake, _ = self.abnorm_discriminator_3d(recovered_abnorm_video.detach())

                            logits_normal2abnorm_image_fake, _ = self.abnorm_discriminator(normal2abnorm_frame_recon.detach())
                            logits_normal2abnorm_video_fake, _ = self.abnorm_discriminator_3d(normal2abnorm_video_recon.detach())

                            d_abnorm_image_loss = self.disc_loss(logits_abnorm_image_real, logits_abnorm_image_fake) + \
                                                  self.disc_loss(logits_abnorm_image_real, logits_normal2abnorm_image_fake)
                                                  # self.calculate_gradient_penalty(self.abnorm_discriminator, abnorm_frames.data, recovered_abnorm_frame.data, args.device) * 10
                                                  # self.calculate_gradient_penalty(self.abnorm_discriminator, abnorm_frames.data, normal2abnorm_frame_recon.data, args.device) * 10
                            d_abnorm_video_loss = self.disc_loss(logits_abnorm_video_real, logits_abnorm_video_fake) + \
                                                  self.disc_loss(logits_abnorm_video_real, logits_normal2abnorm_video_fake)
                                                  # self.calculate_gradient_penalty(self.abnorm_discriminator_3d, abnorm_vids.data, recovered_abnorm_video.data, args.device) * 10
                                                  # self.calculate_gradient_penalty(self.abnorm_discriminator_3d, abnorm_vids.data, normal2abnorm_video_recon.data, args.device) * 10
                            
                            abnorm_discloss = disc_factor * (args.image_gan_weight * d_abnorm_image_loss + args.video_gan_weight * d_abnorm_video_loss)

                            self.opt_disc_abnorm.zero_grad()
                            self.opt_3d_disc_abnorm.zero_grad()
                            abnorm_discloss.backward()
                            self.opt_disc_abnorm.step()
                            self.opt_3d_disc_abnorm.step()

                        if args.local_rank == args.enable_GPUs_id[0]:
                            if args.train_normal:
                                if args.wandb:
                                    wandb.log({'Normal-loss/Image Dis Loss': d_normal_image_loss, 
                                            'Normal-loss/Video Dis Loss': d_normal_video_loss, 
                                            'Normal-loss/OverAll Disc Loss': normal_discloss,},
                                            step = step)
                            if args.train_abnorm:
                                if args.wandb:
                                    wandb.log({'Abnorm-loss/Image Dis Loss': d_abnorm_image_loss, 
                                            'Abnorm-loss/Video Dis Loss': d_abnorm_video_loss,
                                            'Abnorm-loss/OverAll Disc Loss': abnorm_discloss,},
                                            step = step)
                    pbar.update(0)

                    if step % 200 == 0:
                        with torch.no_grad():
                            if args.train_normal:
                                real_fake_normal_images= torch.cat((normal_frames.add(1.0).mul(0.5)[:, :1], normal2abnorm_frame_recon.add(1.0).mul(0.5)[:, :1],
                                                                    identity_frame_B.add(1.0).mul(0.5)[:, :1], recovered_normal_frame.add(1.0).mul(0.5)[:, :1],))
                                vutils.save_image(real_fake_normal_images, os.path.join("results", f"normal_{epoch}_{step}.jpg"), nrow=2)
                            if args.train_abnorm:
                                real_fake_abnorm_images= torch.cat((abnorm_frames.add(1.0).mul(0.5)[:, :1], abnorm2normal_frame_recon.add(1.0).mul(0.5)[:, :1],
                                                                    identity_frame_A.add(1.0).mul(0.5)[:, :1], recovered_abnorm_frame.add(1.0).mul(0.5)[:, :1],))
                                vutils.save_image(real_fake_abnorm_images, os.path.join("results", f"abnorm_{epoch}_{step}.jpg"), nrow=2)

                    if step != 0 and step % 1000 == 0:
                        if args.train_normal:
                            torch.save(self.normal_vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_dual_normal_epoch_{step // 1000}.pt"))
                        if args.train_abnorm:
                            torch.save(self.abnorm_vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_dual_abnorm_epoch_{step // 1000}.pt"))
    
    # Code for WGAN GP
    def calculate_gradient_penalty(self, discriminator, real_images, fake_images, device):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake data]
        if len(real_images.shape) == 4:
            alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
        elif len(real_images.shape) == 5:
            alpha = torch.randn((real_images.size(0), 1, 1, 1, 1), device=device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

        model_interpolates, _ = discriminator(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)

        gradients = gradients[0].view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty * 10

def main(rank, args):

    def wandb_init():
        wandb.init(
            project='CardiacNet',
            entity='CardiacNet Version 1.0',
            name='Enter your name',
            notes='The first version of CardiacNet',
            save_code=True
        )
        wandb.config.update(args)

    try:
        args.local_rank
    except AttributeError:
            args.global_rank = rank
            args.local_rank = args.enable_GPUs_id[rank]
    else:
        if args.distributed:
            args.global_rank = rank
            args.local_rank = args.enable_GPUs_id[rank]

    if args.distributed:
        torch.cuda.set_device(int(args.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.init_method,
                                             world_size=args.world_size,
                                             rank=args.global_rank,
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(args.global_rank), int(args.local_rank)
            ))

        if args.local_rank == args.enable_GPUs_id[0]:
            if args.wandb:
                wandb_init()

    if torch.cuda.is_available(): 
        args.device = torch.device("cuda:{}".format(args.local_rank))
    else: 
        args.device = 'cpu'

    CardiacNet(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CardiacNet")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=(112, 112, 16), help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.7, help='The ratio of masking area in an image (default: 0.75)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='') 
    parser.add_argument('--identity-weight', type=float, default=0.5, help='')
    parser.add_argument('--cycle-weight', type=float, default=10, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=2048)
    parser.add_argument('--n_hiddens', type=int, default=256)
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3, help='The default layer number is 3')
    parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group'])
    parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
    parser.add_argument('--distance_loss', type=str, default='cossimilar', choices=['transport', 'cossimilar'])

    parser.add_argument('--train_normal', type=bool, default=True)
    parser.add_argument('--train_abnorm', type=bool, default=True)
    parser.add_argument('--gan_feat_loss', type=bool, default=False)
    parser.add_argument('--l1_weight', type=float, default=4.0)
    parser.add_argument('--gan_feat_weight', type=float, default=0.1)
    parser.add_argument('--image_gan_weight', type=float, default=1.0)
    parser.add_argument('--video_gan_weight', type=float, default=1.0)
    
    # setting for codebook
    parser.add_argument('--restart_thres', type=float, default=1.0)
    parser.add_argument('--no_random_restart', action='store_true')

    parser.add_argument('--enable_GPUs_id', type=list, default=[4], help='The number and order of the enable gpus')
    parser.add_argument('--wandb', type=bool, default=False, help='Enable Wandb')

    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image'

    # setting distributed configurations
    # args.world_size = 1
    args.world_size = len(args.enable_GPUs_id)
    args.init_method = f"tcp://{get_master_ip()}:{23455}"
    args.distributed = True if args.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and args.distributed:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        # multiple processes have been launched by openmpi
        args.local_rank = args.enable_GPUs_id[0]
        args.global_rank = args.enable_GPUs_id[0]
    
        main(args.local_rank, args)
