# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.codebook import Temporal_Codebook as Codebook
from model.sinkhorn import SinkhornSolver, sinkhorn_rpm, sinkhorn, sinkhorn_cross_batch

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)

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

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv3d(args.latent_dim, args.latent_dim * 2, kernel_size=3, stride=2, bias=True).to(device=args.device)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(args.latent_dim * 2, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        return x


class VQGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.K = 256
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes

        if not hasattr(args, 'padding_type'):
            args.padding_type = 'replicate'
        self.encoder = Encoder(args.n_hiddens, args.image_channels, args.norm_type)
        self.decoder = Decoder(args.n_hiddens, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels

        #self.sinkhorn_loss = SinkhornSolver(epsilon=1e-6, iterations=10, reduction='mean')

        self.ft_queue = F.normalize(torch.randn(256, 784, self.embedding_dim), dim=2)
        self.vq_queue = F.normalize(torch.randn(256, 784, self.embedding_dim), dim=2)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

        self.codebook = Codebook(args.n_codes, args.embedding_dim, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.tempo_pos_embedding = nn.Parameter(torch.randn(1, self.embedding_dim, 16, 1))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, self.embedding_dim, 1, 49))
        nn.init.xavier_uniform_(self.tempo_pos_embedding)
        nn.init.xavier_uniform_(self.space_pos_embedding)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[ptr : ptr + batch_size] = keys.detach()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x, mask = None, templated = False, type = 'N2A'):
        B, C, T, H, W = x.shape
        # get original frames
        frame_idx = torch.randint(0, T, [B]).to(x.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)

        # get masksed frames
        if mask is not None:
            x = x * mask.float()

        z = self.encoder(x)

        b, c, t, h, w = z.shape
        pos_encoding = torch.matmul(self.tempo_pos_embedding, self.space_pos_embedding).view(1, c, t, h, w)
        vq_output = self.codebook(z + pos_encoding)
        pos_weight = torch.matmul(self.tempo_pos_embedding.view(1, c, -1).transpose(-1,-2), self.space_pos_embedding.view(1, c, -1)).view(-1)
        # where 1e-9 is eps, in case divide by zero
        pos_weight = pos_weight - torch.min(pos_weight) + 1e-9
        pos_weight = pos_weight / torch.max(pos_weight)

        if templated:
            optimal_trans_tf_loss = 0
            optimal_trans_vq_loss = 0

            ft_code = z.view(b, c, -1).transpose(-1, -2)
            vq_code = vq_output['embeddings'].view(b, c, -1).transpose(-1, -2)

            ft_current = ft_code
            vq_current = vq_code
            # if type == 'N2A':
            #     ft_current = ft_code[:self.args.batch_size]
            #     vq_current = vq_code[:self.args.batch_size]
            # if type == 'A2N':
            #     ft_current = ft_code[self.args.batch_size:]
            #     vq_current = vq_code[self.args.batch_size:]
            
            self._dequeue_and_enqueue(ft_current, self.ft_queue)
            self._dequeue_and_enqueue(vq_current, self.vq_queue)

            # deformation_tf_centroid = torch.mean(self.ft_queue, dim = 0).to(z.device)
            deformation_vq_centroid = torch.mean(self.vq_queue, dim = 0).to(z.device)
            deformation_vq_m = torch.mean(self.vq_queue, dim = 1).to(z.device)
            w_ = pos_weight.detach()

            # optimal_trans_tf_loss = sinkhorn_cross_batch(ft_current.contiguous(), deformation_tf_centroid.repeat(self.args.batch_size, 1, 1).detach().contiguous(), w_x=w_, w_y=w_)[0]
            optimal_trans_vq_loss = sinkhorn_cross_batch(vq_current.contiguous(), deformation_vq_centroid.repeat(vq_current.shape[0], 1, 1).detach().contiguous(), w_x=w_, w_y=w_)[0]
            
            # optimal_trans_tf_loss = 0.01 * (optimal_trans_tf_loss / b)
            optimal_trans_vq_loss = 0.01 * (optimal_trans_vq_loss / b)

        commitment_loss = vq_output['commitment_loss']
        x_recon = self.decoder(vq_output['embeddings'])
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if templated:
            return x_recon, frames, frames_recon, (self.ft_queue, deformation_vq_m.unsqueeze(0), optimal_trans_tf_loss, optimal_trans_vq_loss, w_, commitment_loss), z
        else:
            return x_recon, frames, frames_recon, (vq_output['embeddings'], None, commitment_loss), z


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, image_channel=3, norm_type='group'):
        super().__init__()

        channels = [64, 128, 128, 256, 256]
        self.conv_blocks = nn.ModuleList()

        self.conv_first = nn.Conv3d(image_channel, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        for i in range(len(channels)-1):
            block = nn.Module()
            in_channels = channels[i]
            out_channels = channels[i + 1]
            block.down = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            nn.Conv3d(out_channels, n_hiddens, kernel_size=3, stride=1, padding=1, bias=True),
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)

        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, image_channel, norm_type='group'):
        super().__init__()

        channels = [256, 256, 128, 128, 64]

        self.final_block = nn.Sequential(
            Normalize(channels[0], norm_type),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            block = nn.Module()

            in_channels = channels[i]
            out_channels = channels[i + 1]

            block.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            block.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            #block.non_local = NonLocalBlock(out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)

        self.norm_last = Normalize(out_channels, norm_type)
        self.conv_last = nn.Conv3d(out_channels, image_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.conv(h)
            h = block.res1(h)
            # h = block.non_local(h)
            h = block.res2(h)
        h = self.norm_last(h)
        h = silu(h)
        h = self.conv_last(h)

        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, norm_type):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels

        self.gn = Normalize(in_channels, norm_type)
        self.q = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape

        q = q.reshape(b, c, t*h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, t*h*w)
        v = v.reshape(b, c, t*h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, t, h, w)

        return x + A

    
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _
