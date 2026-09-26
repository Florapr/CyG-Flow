"""CyG-Flow base: feature backbone helpers + normalizing-flow stack + multi-scale fusion."""

import os
import sys

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


def subnet_conv_func(kernel_size, hidden_ratio, activation="relu"):
    activation = str(activation).lower()
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "silu":
        act_layer = nn.SiLU
    elif activation == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError("flow_activation must be one of ['relu', 'silu', 'gelu']")

    def subnet_conv(in_channels, out_channels):
        hidden_channels = max(int(in_channels * hidden_ratio), 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            act_layer(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_cyg_flow(
    input_chw,
    conv3x3_only,
    hidden_ratio,
    flow_steps,
    clamp=2.0,
    flow_activation="relu",
):
    nodes = Ff.SequenceINN(*input_chw)
    kernel_size = 3 if not conv3x3_only else 3
    for _ in range(flow_steps):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(
                kernel_size,
                hidden_ratio,
                activation=flow_activation,
            ),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class CyGFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
        vssm_ckpt=None,
        vssm_out_indices=None,
        fusion_weights=None,
        fusion_learnable=False,
        fusion_init=None,
        flow_activation="relu",
    ):
        super(CyGFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if backbone_name in [const.BACKBONE_RESNET18, const.BACKBONE_VSSM_SMALL]:
            vmamba_models_dir = os.path.join(
                os.path.dirname(__file__), "VMamba", "classification", "models"
            )
            if vmamba_models_dir not in sys.path:
                sys.path.insert(0, vmamba_models_dir)
            from VMamba.classification.models.vmamba import Backbone_VSSM

            if vssm_ckpt is None:
                vssm_ckpt = os.path.join(
                    os.path.dirname(__file__),
                    "vim_small_midclstok",
                    "vssm_small_0229_ckpt_epoch_222.pth",
                )

            if vssm_out_indices is None:
                vssm_out_indices = (0, 1, 2)
            vssm_out_indices = tuple(int(i) for i in vssm_out_indices)
            if (
                len(vssm_out_indices) < 1
                or len(vssm_out_indices) > 4
                or any(i < 0 or i > 3 for i in vssm_out_indices)
            ):
                raise ValueError(
                    "vssm_out_indices must contain 1~4 stage indices in [0,1,2,3]"
                )
            if len(set(vssm_out_indices)) != len(vssm_out_indices):
                raise ValueError("vssm_out_indices must not contain duplicated stages")

            stage_channels = {0: 96, 1: 192, 2: 384, 3: 768}
            stage_scales = {0: 4, 1: 8, 2: 16, 3: 32}

            self.feature_extractor = Backbone_VSSM(
                out_indices=vssm_out_indices,
                pretrained=vssm_ckpt,
                norm_layer="ln2d",
                depths=[2, 2, 15, 2],
                dims=96,
                drop_path_rate=0.3,
                patch_size=4,
                in_chans=3,
                num_classes=1000,
                ssm_d_state=1,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer="silu",
                ssm_conv=3,
                ssm_conv_bias=False,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v05_noz",
                mlp_ratio=4.0,
                mlp_act_layer="gelu",
                mlp_drop_rate=0.0,
                gmlp=False,
                patch_norm=True,
                downsample_version="v3",
                patchembed_version="v2",
                use_checkpoint=False,
                posembed=False,
                imgsize=224,
            )
            channels = [stage_channels[i] for i in vssm_out_indices]
            scales = [stage_scales[i] for i in vssm_out_indices]
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )
        elif backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_cyg_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    flow_activation=flow_activation,
                )
            )

        self.fusion_learnable = bool(fusion_learnable)
        self.fusion_logits = None
        if self.fusion_learnable:
            n = len(channels)
            if fusion_init is not None:
                if len(fusion_init) != n:
                    raise ValueError(
                        "fusion_init length {} must match number of scales {}".format(
                            len(fusion_init), n
                        )
                    )
                t = torch.tensor([float(w) for w in fusion_init], dtype=torch.float32)
                t = t / t.sum()
                logits = torch.log(t + 1e-8)
            else:
                logits = torch.zeros(n, dtype=torch.float32)
            self.fusion_logits = nn.Parameter(logits)
        else:
            if fusion_weights is None:
                # Default WMF coefficients used in the paper for three scales.
                if len(channels) == 3:
                    weights = [0.2, 0.3, 0.5]
                else:
                    weights = [1.0 / len(channels)] * len(channels)
            else:
                if len(fusion_weights) != len(channels):
                    raise ValueError(
                        "fusion_weights length {} must match number of scales {}".format(
                            len(fusion_weights), len(channels)
                        )
                    )
                weights = [float(w) for w in fusion_weights]
                s = sum(weights)
                if s <= 0:
                    raise ValueError("fusion_weights sum must be > 0")
                weights = [w / s for w in weights]
            self.register_buffer(
                "fusion_weights", torch.tensor(weights, dtype=torch.float32)
            )
        self.input_size = input_size

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                prob_up = F.interpolate(
                    prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(-prob_up)

            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            if self.fusion_learnable:
                w = F.softmax(self.fusion_logits, dim=0).to(anomaly_map_list.device)
                weights = w.view(1, 1, 1, 1, -1)
            else:
                weights = self.fusion_weights.to(anomaly_map_list.device).view(
                    1, 1, 1, 1, -1
                )
            anomaly_map = torch.sum(anomaly_map_list * weights, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret
