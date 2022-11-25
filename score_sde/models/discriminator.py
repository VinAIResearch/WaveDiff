# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from . import dense_layer, layers, up_or_down_sampling


dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim=128,
        downsample=False,
        act=nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out


class Discriminator_small(nn.Module):
    """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""

    def __init__(self, nc=3, ngf=64, t_emb_dim=128, act=nn.LeakyReLU(0.2), patch_size=1, use_local_loss=False, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.use_local_loss = use_local_loss
        nc = nc * self.patch_size * self.patch_size
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        # Encoding layers where the resolution decreases
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 2, t_emb_dim=t_emb_dim, act=act)

        self.conv2 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4 = None
        if num_layers >= 4:
            self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1, init_scale=0.)
        self.end_linear = dense(ngf * 8, 1)
        if use_local_loss:
            self.local_end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        x = rearrange(x, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        x_t = rearrange(x_t, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)

        h2 = self.conv2(h1, t_embed)

        h3 = self.conv3(h2, t_embed)

        if self.conv4 is not None:
            h3 = self.conv4(h3, t_embed)
        out = h3

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        # if self.patch_size == 1:
        # 	out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        # else:
        # 	out = out.view(out.shape[0], out.shape[1], -1).permute(0,2,1)
        out = out.view(out.shape[0], out.shape[1], -1)
        t = out
        out = self.end_linear(out.sum(2))
        if self.use_local_loss:
            out2 = self.local_end_linear(t.permute(0, 2, 1))
            return (out, out2)

        return out


class Discriminator_large(nn.Module):
    """A time-dependent discriminator for large images (CelebA, LSUN)."""

    def __init__(self, nc=1, ngf=32, t_emb_dim=128, act=nn.LeakyReLU(0.2), patch_size=1, use_local_loss=False, num_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.use_local_loss = use_local_loss

        nc = nc * self.patch_size * self.patch_size
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv2 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4, self.conv5, self.conv6 = None, None, None
        if num_layers >= 4:
            self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        if num_layers >= 5:
            self.conv5 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        if num_layers >= 6:
            self.conv6 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)
        if use_local_loss:
            self.local_end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        x = rearrange(x, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        x_t = rearrange(x_t, "n c (h p1) (w p2) -> n (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)

        h = self.conv2(h, t_embed)

        h = self.conv3(h, t_embed)
        if self.conv4 is not None:
            h = self.conv4(h, t_embed)
        if self.conv5 is not None:
            h = self.conv5(h, t_embed)
        if self.conv6 is not None:
            h = self.conv6(h, t_embed)
        out = h

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        # if self.patch_size == 1:
        # 	out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        # else:
        # 	out = out.view(out.shape[0], out.shape[1], -1).permute(0,2,1)

        out = out.view(out.shape[0], out.shape[1], -1)
        t = out
        out = self.end_linear(out.sum(2))
        if self.use_local_loss:
            out2 = self.local_end_linear(t.permute(0, 2, 1))
            return (out, out2)

        return out
