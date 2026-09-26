"""CyG-Flow: VMamba (vssm_small) + normalizing flow."""

import os
import sys

import torch
import torch.nn as nn

import constants as const
import flow as _base


_SELECTIVE_SCAN_TORCH_PATCH_DONE = False


def _parse_gpu_arg_from_argv():
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            try:
                return int(argv[i + 1].strip())
            except ValueError:
                return None
        if a.startswith("--gpu="):
            try:
                return int(a.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _ensure_fvcore_for_vmamba():
    try:
        import fvcore.nn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "VMamba (vssm_small) requires fvcore. Install with: pip install fvcore"
        ) from e


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _should_use_torch_selective_scan_fallback():
    if _env_truthy("CYG_VM_FORCE_TORCH_SELECTIVE_SCAN"):
        return True, "CYG_VM_FORCE_TORCH_SELECTIVE_SCAN"
    if _env_truthy("CYG_VM_DISABLE_AUTO_TORCH_SELECTIVE_SCAN"):
        return False, None
    if not torch.cuda.is_available():
        return False, None

    forced_gpu = _parse_gpu_arg_from_argv()
    if forced_gpu is not None and forced_gpu < 0:
        return False, None

    if forced_gpu is not None:
        n = torch.cuda.device_count()
        if forced_gpu >= n:
            return False, None
        major, minor = torch.cuda.get_device_capability(forced_gpu)
        if (major, minor) < (8, 9):
            return True, "auto (--gpu {} is sm_{}.{})".format(
                forced_gpu, major, minor
            )
        return False, None

    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if (major, minor) < (8, 9):
            return True, "auto (visible device {} is sm_{}.{})".format(
                i, major, minor
            )
    return False, None


def _install_selective_scan_torch_workaround():
    global _SELECTIVE_SCAN_TORCH_PATCH_DONE
    if _SELECTIVE_SCAN_TORCH_PATCH_DONE:
        return
    use_torch, reason = _should_use_torch_selective_scan_fallback()
    if not use_torch:
        return

    import VMamba.classification.models.csms6s as csms6s

    def _wrapped(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=True,
        oflex=True,
        backend=None,
    ):
        return csms6s.selective_scan_fn_torch(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            oflex=oflex,
        )

    csms6s.selective_scan_fn = _wrapped
    _SELECTIVE_SCAN_TORCH_PATCH_DONE = True
    print(
        "[CyG-Flow] Using torch selective_scan fallback ({})".format(reason),
        flush=True,
    )


def nf_cyg_flow_vmamba(
    input_chw,
    conv3x3_only,
    hidden_ratio,
    flow_steps,
    clamp=2.0,
    flow_activation="relu",
):
    """Build the normalizing-flow stack used by ``CyGFlow``."""
    return _base.nf_cyg_flow(
        input_chw,
        conv3x3_only=conv3x3_only,
        hidden_ratio=hidden_ratio,
        flow_steps=flow_steps,
        clamp=clamp,
        flow_activation=flow_activation,
    )


class CyGFlow(_base.CyGFlow):
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
        **kwargs,
    ):
        del kwargs
        if str(backbone_name).strip() != const.BACKBONE_VSSM_SMALL:
            raise ValueError(
                "CyG-Flow only supports backbone_name=vssm_small, got {!r}".format(
                    backbone_name
                )
            )

        _ensure_fvcore_for_vmamba()
        _install_selective_scan_torch_workaround()

        _base.nn.Module.__init__(self)
        self.input_size = input_size

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

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_cyg_flow_vmamba(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    flow_activation=flow_activation,
                )
            )

        self.channels = channels
        self.scales = scales
        self.fusion_learnable = bool(fusion_learnable)
        self.fusion_logits = None
        if self.fusion_learnable:
            n = len(channels)
            if fusion_init is not None:
                if len(fusion_init) != n:
                    raise ValueError(
                        f"fusion_init length {len(fusion_init)} must match scales {n}"
                    )
                init = _base.torch.tensor(fusion_init, dtype=_base.torch.float32)
            else:
                init = _base.torch.ones(n, dtype=_base.torch.float32)
            init = init / init.sum()
            self.fusion_logits = nn.Parameter(init.log())
        else:
            if fusion_weights is None:
                fusion_weights = [0.2, 0.3, 0.5] if len(channels) == 3 else (
                    [1.0 / len(channels)] * len(channels)
                )
            if len(fusion_weights) != len(channels):
                raise ValueError(
                    f"fusion_weights length {len(fusion_weights)} must match scales {len(channels)}"
                )
            fw = _base.torch.tensor(fusion_weights, dtype=_base.torch.float32)
            fw = fw / fw.sum()
            self.register_buffer("fusion_weights", fw)

    forward = _base.CyGFlow.forward


for _name in dir(_base):
    if _name.startswith("__") or _name == "CyGFlow":
        continue
    globals()[_name] = getattr(_base, _name)
