from transformers import AutoModel
from sglang.srt.configs.livln_config import VideoLlavaQwenConfig
import torch


from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from typing import List, Optional, Tuple, Union, Iterable

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.attention.vision import VisionAttention

import torch.nn as nn
import torchvision.transforms
from transformers import Qwen2Config, AutoConfig
import re
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from timm.models._manipulate import checkpoint_seq, named_apply
import torch.nn.functional as F
from functools import partial
import warnings
import math
from abc import ABC, abstractmethod

from sglang.srt.models.qwen2 import Qwen2Model, Qwen2ForCausalLM


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_VIDEO_TOKEN = "<video>"


from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    Any,
)

def dense_connector_dci(image_forward_outs):
    image_features_1 = []
    image_features_2 = []
    for i in range(0, 12):
        image_features_1.append(image_forward_outs[i])
    image_features_1 = torch.stack(image_features_1, dim=0)
    image_features_1 = torch.sum(image_features_1, dim=0) / 12
    for i in range(12, 24):
        image_features_2.append(image_forward_outs[i])
    image_features_2 = torch.stack(image_features_2, dim=0)
    image_features_2 = torch.sum(image_features_2, dim=0) / 12
    return torch.cat([image_features_1, image_features_2], dim=-1)

def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
        
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtype = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtype)

def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim ** -0.5)
    
from timm.layers import (
    AttentionPoolLatent,
    DropPath,
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    resample_abs_pos_embed,
)

SigLIP_MODEL_CONFIG = {
    "siglip_large_with_global_token_patch16_512_1280": {
        "image_size": (512, 1280),
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
        "return_with_global_token": True,
    },
}

@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    return_with_global_token: bool = False
    select_layer: int = -1  # 取特定的单层视觉特征
    select_compound_layer: Union[int, dict, None] = None  # 视觉融合特征， int表示取倒数前n层特征，dict表示取特定的特征层

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def as_is(x):
    return x

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = True
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else as_is
        self.k_norm = norm_layer(self.head_dim) if qk_norm else as_is
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else as_is

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = VisionAttention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_norm=qk_norm,
        #     attn_drop=attn_drop,
        #     proj_drop=proj_drop,
        #     norm_layer=norm_layer,
        # )
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            use_context_forward=False,
            softmax_in_single_precision=False,
            dropout=attn_drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else as_is
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else as_is

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else as_is
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else as_is

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
        return_with_global_token: bool = False,
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        # norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = get_act_layer(act_layer) or nn.GELU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        self.return_with_global_token = return_with_global_token
        if self.return_with_global_token:
            assert global_pool == 'map' and ignore_head, 'siglip uses the global_pooling to generate global token'
            print('[INFO]: siglip, return with global_token, i.e. [B, 1 + N, D]')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = (
            no_embed_class  # don't embed prefix positions (includes reg)
        )
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        # head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert (
                    False
                ), "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(
            range(num_blocks - n, num_blocks) if isinstance(n, int) else n
        )

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0: self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.return_with_global_token:
            x_cls = self.attn_pool(x)
            x = torch.cat([x_cls[:, None, :], x], dim=1)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x
  

def dci_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return torch.cat([result[-1], dense_connector_dci(result)], dim=-1) if isinstance(result, tuple) else result

    return wrapper

def mean_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return torch.stack(result, dim=0).mean(dim=0) if isinstance(result, tuple) else result

    return wrapper

def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = 384,
    select_layer: int = -1,
    ckpt_path: str = "",
    **kwargs,
):
    assert (
        model_name in SigLIP_MODEL_CONFIG.keys()
    ), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    select_layer = getattr(vision_cfg, 'select_layer', select_layer)
    select_compound_layer = getattr(vision_cfg, 'select_compound_layer', None)

    if not select_compound_layer:
        print(f'[INFO]: Select Vision embedding from {select_layer} layer.')
    else:
        select_layer = -1

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    model = VisionTransformer(
        img_size=image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        weight_init=kwargs.get("weight_init", "skip"),
        return_with_global_token=vision_cfg.return_with_global_token,
        num_classes=0,
    )

    if select_compound_layer and 'compound' in model_name and "dci" in model_name:
        note = 'last ' + str(select_compound_layer) if isinstance(select_compound_layer, int) else select_compound_layer
        print(f'[INFO]: Select compound vision embedding from the {note} layer.')

        model.forward = dci_tuple(
            partial(model.get_intermediate_layers, n=select_compound_layer)
        )
    elif select_compound_layer and 'compound' in model_name:
        note = 'last ' + str(select_compound_layer) if isinstance(select_compound_layer, int) else select_compound_layer
        print(f'[INFO]: Select compound vision embedding from the {note} layer.')

        model.forward = mean_tuple(
            partial(model.get_intermediate_layers, n=select_compound_layer)
        )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith("visual.trunk."):
                new_key = key[13:]
                new_state_dict[new_key] = state_dict[key]
        if 'pos_embed' in new_state_dict.keys():
            if new_state_dict['pos_embed'].shape[1] != model.pos_embed.shape[1]:
                print("pos_embed shape in model not match pretrain model, start to resample, new grid size: ",
                      model.patch_embed.grid_size)
                # To resize pos embedding when using model at different size from pretrained weights
                num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model,
                                                                                              'num_prefix_tokens', 1)
                v = resample_abs_pos_embed(
                    new_state_dict['pos_embed'],
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    # interpolation=interpolation,
                    # antialias=antialias,
                    verbose=True,
                )
                new_state_dict['pos_embed'] = v

        incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
        # print("#" * 20)
        print(
            f"SigLIP-ViT restores from {ckpt_path},\n"
            f"\tincompatible_keys:', {incompatible_keys}."
        )

    return model


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        downsample: bool = False,
        use_high_res_cam120_area: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        if "sam_hd" in kwargs.keys():
            vision_tower_params["sam_hd"] = kwargs["sam_hd"]

        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params
        )
        if hasattr(self.vision_tower, "embed_dim"):
            self.hidden_size = self.vision_tower.embed_dim
        else:
            self.hidden_size = 1024
        self.downsample = downsample

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm
        if isinstance(image_size, tuple):
            assert self.image_norm == None
            # # self.image_processor = SigLIPImageProcessor(image_size=image_size, do_normalize=True, use_high_res_cam120_area=use_high_res_cam120_area, processor_class='SigLIPImageProcessor')
            # # self.image_processor.save_pretrained("/lpai/volumes/ad-parking-vol-ga/linyuan/7B-RL-models/0325-mindad-7b-sft-10ep-9w/checkpoint/0325-baidu-sftcoldstart-9w-10ep")
            # self.image_processor = AutoProcessor.from_pretrained("/lpai/volumes/ad-parking-vol-ga/linyuan/7B-RL-models/0325-mindad-7b-sft-10ep-9w/checkpoint/0325-baidu-sftcoldstart-9w-10ep")

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()
        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    def forward(self, images):
        if self.image_norm is not None:
            images = self.image_norm(images)
        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)

        return image_features

    def load_model(self):
        self.is_loaded = True

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_vision_tower(vision_tower_cfg, **kwargs):
    siglip_model_name = getattr(vision_tower_cfg, "siglip", None)
    if siglip_model_name:
        if siglip_model_name == 'siglip':
            siglip_model_name = 'siglip_large_patch16_384'
        print("[INFO]: using siglip encoder", siglip_model_name)
        image_size = None
        pattern = r"_(\d+)(?:_(\d+))?$"
        match = re.search(pattern, siglip_model_name)
        if match:
            image_size = [int(num) for num in match.groups() if num is not None]
        else:
            ValueError(f'Unknown siglip vision tower: {siglip_model_name}')
        if len(image_size) == 1:
            image_size = tuple([image_size[0], image_size[0]])
        else:
            image_size = tuple(image_size)
        siglip_downsample = getattr(vision_tower_cfg, "siglip_downsample", False)
        low_res_cfg = dict(
            model_name=siglip_model_name,
            select_feature="same",
            image_size=image_size,
            pixel_mean=None,
            pixel_std=None,
            select_layer=-1,
            ckpt_path=vision_tower_cfg.siglip_pretrain_ckpt,
            downsample=siglip_downsample,
            use_high_res_cam120_area=getattr(vision_tower_cfg, 'use_high_res_cam120_area', False)
        )
        return CLIPVisionTower(**low_res_cfg)

def build_vision_mixer(config, **kwargs):
    return nn.Identity()

class TemporalSEBlock(nn.Module):
    """modified from SE Block, weighted pooling at temporal dimension"""
    def __init__(self, frame_num=8, dim=256):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(frame_num, dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim, frame_num, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, t, n, d = x.shape
        y = self.squeeze(x.view(bs, t, -1)).view(bs, t)
        y = self.excitation(y).view(bs, t, 1, 1)
        return x * y.expand_as(x)

class GloableLocalFeatureSelector(nn.Module):
    def __init__(self, in_channel, extend_token_num=120, final_h = 12, final_w = 30):
        super().__init__()
        self.in_channel = in_channel
        self.extend_token_num = extend_token_num
        self.final_h = final_h
        self.final_w = final_w

    def cal_cross_attn_score(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores = scores.softmax(dim=-1)
        return scores

    def downsample_feat(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).view(-1, c, h, w)
        # x = nn.functional.adaptive_avg_pool2d(x, output_size=(self.final_h, self.final_w))
        x = F.interpolate(x, size=(self.final_h, self.final_w), mode='bilinear', align_corners=False)
        x = x.view(b, t, c, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        return x

    def forward(self, x: torch.Tensor, cls_tokens: torch.Tensor):

        b, c, t, h, w = x.shape # LC ([1, 1024, 1, 12, 30])
        if h == self.final_h and w == self.final_w:
            x = x.permute(0, 2, 3, 4, 1).view(b, t, h*w, c) # [b, c, t, h, w] -> [b, t, h, w, c] -> [b, t, h*w, c]
        else:
            x = self.downsample_feat(x) # [b, c, t, h, w] -> [b, c, t, h/2, w/2]
            x = x.permute(0, 2, 3, 4, 1).view(b, t, self.final_h * self.final_w, c) # [b, c, t, h, w] -> [b, t, h, w, c] -> [b, t, h*w, c]

        # for single frame image, directly return the global tokens
        if t == 1:
            global_tokens = x[:, 0, :, :] # [b, n, c]
            final_tokens = torch.cat([cls_tokens, global_tokens], dim=1) # [b ,n+1, c]
            return final_tokens

        attn_score_list = []
        for i in range(t):
            ori_img_patch = x[:, i, :, :] # [b, t, n, c] -> [b, n, c]
            cls_token = cls_tokens[:, i, :].unsqueeze(dim=1) # [b, t, c] -> [b, c] -> [b, 1, c]
            with torch.no_grad():
                attn_output_weights = self.cal_cross_attn_score(cls_token, ori_img_patch) # [b, 1, n]
                # LC: the scores is only for ranking, the max op is not necessary
                atten_score_frame = attn_output_weights / attn_output_weights.max() # [b, 1, n]
                attn_score_list.append(atten_score_frame)
        attn_scores = torch.cat(attn_score_list, dim=1) # [b, t, n]

        # LC: get top-k scores and indices
        _, top_k_indices = torch.topk(attn_scores, self.extend_token_num)

        # LC get local tokens from image patchs
        batch_indices = torch.arange(b).view(b, 1, 1).expand(-1, t, self.extend_token_num) # b index
        time_indices = torch.arange(t).view(1, t, 1).expand(b, -1, self.extend_token_num) # t index
        local_tokens = x[batch_indices, time_indices, top_k_indices, :] # [b, t, num_local_tokens, c]

        # LC: concat cls_token, local_tokens, global_tokens
        sec_frame_global_tokens = x[:, 1, :, :] # [b, n, c]
        fir_frame_local_tokens = local_tokens[:, 0, :, :] # [b, n, c]
        fir_frame_cls_token = cls_tokens[:, 0, :].unsqueeze(dim=1)
        sec_frame_cls_token = cls_tokens[:, 1, :].unsqueeze(dim=1)
        final_tokens = torch.cat([fir_frame_cls_token, fir_frame_local_tokens, sec_frame_cls_token, sec_frame_global_tokens], dim=1) # [b, n, c]

        return final_tokens
    
class Conv3DDownLayer(nn.Module):
    def __init__(self, in_channel, out_channel, feature_h, feature_w, kernel_size, stride,
                 num_frames, projector_type, use_temporal_se_weight=False, use_temporal_max_pooling=False,
                 use_internvideo=False, use_peg_standard_conv=False, feature_selector_topk_ratio=0.5) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.cnn_dwn = nn.Conv3d(
            self.in_channel,
            self.out_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 1, 1),
            bias=False,
        )
        groups = 1 if use_peg_standard_conv else self.out_channel
        self.peg = nn.Sequential(
            nn.Conv3d(self.out_channel,
                      self.out_channel,
                      kernel_size=(1, 3, 3),
                      stride=(1, 1, 1),
                      padding=(0, 1, 1),
                      bias=True,
                      groups=groups)
        )
        self.projector_type = projector_type
        if projector_type == "st_cdpnet_gl_feature_selector":
            extend_token_num = 120
            print("[INFO]: using st_cdpnet_gloable_local_feature_selector, extended token num:{}".format(extend_token_num))
            self.temporal_pooling = GloableLocalFeatureSelector(self.out_channel, extend_token_num)
            self.use_post_norm = True
            self.post_norm = nn.LayerNorm(self.in_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4]

        if x.ndim == 3:
            x = x.unsqueeze(1)
        b, t, num_tokens, c = x.shape
        # 第一个token是否为cls token
        return_with_cls_token = num_tokens % 2
        if return_with_cls_token:
            num_tokens -= 1
            x_cls, x = x[:, :, 0, :], x[:, :, 1:, :]
        h, w = self.feature_h, self.feature_w
        x = x.permute(0, 3, 1, 2).view(b, c, t, h, w)
        cnn_feat = self.cnn_dwn(x)  # [b, c, t, h/2, w/2]
        x = self.peg(cnn_feat) + cnn_feat  # [b, c, t, h/2, w/2]
        if self.temporal_pooling:
            if return_with_cls_token: x = self.temporal_pooling(x, x_cls)  # [b, c, 1, h/2, w/2]
            else: x = self.temporal_pooling(x)  # [b, c, 1, h/2, w/2]
        if self.projector_type not in ["st_cdpnet_gl_feature_selector", "st_cdpnet_camcrop"]:
            x = x.flatten(2).transpose(1, 2)  # [b, num_tokens, c]
        if self.use_post_norm:
            x = self.post_norm(x)
        return x

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class STCDPNetProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        use_internvideo = False
        projector_type = getattr(config, "mm_projector_type", "st_cdpnet_flatten")
        num_frames = getattr(config, 'num_frames', 8)
        use_temporal_se_weight = getattr(config, 'use_temporal_se_weight', False)
        use_temporal_max_pooling = getattr(config, 'use_temporal_max_pooling', False)
        feature_selector_topk_ratio = getattr(config, 'feature_selector_topk_ratio', 0.5)
        
        feature_h, feature_w, kernel_size, stride = 32, 80, (1, 3, 3), (1, 2, 2)
        use_peg_standard_conv = projector_type == 'st_cdpnet_standard_conv'
        self.dwn = Conv3DDownLayer(inc, inc, feature_h, feature_w, kernel_size, stride, num_frames,
                                    projector_type, use_temporal_se_weight, use_temporal_max_pooling,
                                    use_internvideo, use_peg_standard_conv, feature_selector_topk_ratio)


        self.mlp = FeatureIRLayer(inc, ouc)
        print(f'======= using spatial temporal cdpnet, single frame token num: {feature_h * feature_w}')

    def forward(self, x):
        x = self.dwn(x)
        x = self.mlp(x)
        return x
    
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if 'st_cdpnet' in projector_type or 'st_2d_cdpnet_flatten' in projector_type:
        return STCDPNetProjector(config)
    
class VideoLlavaMetaModel:
    def __init__(self, config):
        super(VideoLlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_mixer = build_vision_mixer(config)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

class VideoLlavaQwenConfig(Qwen2Config):
    model_type = "video_llava_qwen"


class VideoLlavaQwenModel(VideoLlavaMetaModel, Qwen2Model):
    config_class = VideoLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(VideoLlavaQwenModel, self).__init__(config)

class VideoLlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        images = images.to(dtype=self.get_model().mm_projector.mlp.mlp[0].weight.dtype)
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_mixer(image_features)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, images, forward_batch
    ):
        batch_size = forward_batch.batch_size
        images_minibatch = images.reshape(batch_size, -1, *images.shape[1:])
        # split input_ids into minibatch give by seq_lens
        cum_seq_lens = forward_batch.seq_lens.cumsum(dim=0).cpu()
        split_input_ids = torch.tensor_split(input_ids, cum_seq_lens, dim=0)
        if getattr(images_minibatch, 'ndim', 0) == 4:  # batch consists of images, [b, c, h, w]
            image_features_minibatch = self.encode_images(images_minibatch)  # [b, l, c]

        input_embeds = []
        for batch_idx in range(batch_size):
            cur_input_ids = split_input_ids[batch_idx]
            new_input_embeds = []
            pad_values = forward_batch.mm_inputs[batch_idx].pad_values
            image_token_idx = torch.where(cur_input_ids == pad_values[0])[0].tolist()
            new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_idx[0]]))
            if images is not None:
                new_input_embeds.append(image_features_minibatch[batch_idx].to(device=torch.cuda.current_device()))
            new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_idx[-1]+1:]))
            input_embeds.append(torch.cat(new_input_embeds, dim=0))
        del images_minibatch
        del images
        del image_features_minibatch
        input_embeds = torch.cat(input_embeds, dim=0)
        return input_embeds


class VideoLlavaQwenForCausalLM(Qwen2ForCausalLM, VideoLlavaMetaForCausalLM):
    config_class = VideoLlavaQwenConfig

    def __init__(self,
                 config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        Qwen2ForCausalLM.__init__(self, config, quant_config, prefix=prefix)
        VideoLlavaMetaForCausalLM.__init__(self)
        # super().__init__(config, quant_config, prefix=prefix)
        
        if getattr(config, 'siglip_pretrain_ckpt', ""):
            print('[INFO]: Do not load siglip ckpt twice during the `from_pretrained` mode.')
            config.siglip_pretrain_ckpt = ""
        self.config = config
        
        self.model = VideoLlavaQwenModel(config)
        self.vocab_size = config.vocab_size

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):

        if (
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            images = forward_batch.merge_mm_inputs()
            images = images.pixel_values.to(device=torch.cuda.current_device(), non_blocking=True)
            input_ids = input_ids.to(device=torch.cuda.current_device(), non_blocking=True)
            
            inputs_embeds = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                images=images,
                forward_batch=forward_batch)
            # once used, mm_inputs is useless
            # just being defensive here
            forward_batch.mm_inputs = None
        else:
            inputs_embeds =  self.get_model().embed_tokens(input_ids)
        output = super().forward(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
        )
        return output

    def get_input_embeddings(self):
        return self.get_model().embed_tokens
    

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        pad_values = image_inputs.pad_values
        # token_id for image token in input_ids is IMAGE_TOKEN_INDEX, replace it with pad_values
        # input_ids is a list of int
        for i in range(len(input_ids)):
            if input_ids[i] == IMAGE_TOKEN_INDEX:
                input_ids[i] = pad_values[0]
        return input_ids


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # if "rotary_emb.inv_freq" in name or "projector" in name:
            #     continue
            # if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            #     # Models trained using ColossalAI may include these tensors in
            #     # the checkpoint. Skip them.
            #     continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # adapt to VisionAttention
            if "vision_tower.vision_tower" in name:
                name = name.replace("attn.qkv", "attn.qkv_proj")

            # skip generation sub model
            if "gen" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # replace the name and load with customized loader
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                # # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

AutoConfig.register("video_llava_qwen", VideoLlavaQwenConfig)
AutoModel.register(config_class=VideoLlavaQwenConfig, model_class=VideoLlavaQwenForCausalLM)
EntryClass = [VideoLlavaQwenForCausalLM]