# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import List, Optional, Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from mamba_ssm import Mamba2

from nets_works.FGM import FGM
from nets_works.HSCF import HilbertSFCMambaLayer
from nets_works.utils import LargeKernelConv, MlpChannel,GSC

from nets_works.gdysample import GDySampleUpBlock3D



class MambaEncoder(nn.Module):
    """
    MambaEncoder
    - use_kan=False: 所有 stage 使用 MLP
    - use_kan=True : stage 0/1 使用 MLP，stage 2/3 使用 KAN（固定规则）
    """
    def __init__(
        self,
        in_chans: int = 1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        out_indices=[0, 1, 2, 3],
        mamba_cls=Mamba2,
        mamba_kwargs_per_stage: list | None = None,
        use_fgm: bool = True,
        use_fgm_stage: list[int] = [0,1,2, 3],
        use_rope: bool = True,
        use_kan: bool = False,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.depths = depths
        self.dims = dims
        self.out_indices = out_indices
        self.use_fgm = use_fgm
        self.use_fgm_stage = use_fgm_stage
        self.use_rope = use_rope
        self.use_kan = use_kan

        # -------------------------
        # Downsample layers
        # -------------------------
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1)
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
            )

        # -------------------------
        # FGM per stage
        # -------------------------
        self.fgms = nn.ModuleList()
        for i, d in enumerate(dims):
            if self.use_fgm and i in self.use_fgm_stage:
                self.fgms.append(FGM(d, layer_idx=i))
            else:
                self.fgms.append(nn.Identity())

        # -------------------------
        # Shallow convs (stage 0-1)
        # -------------------------
        self.shallow_convs = nn.ModuleList()

        for i in range(2):
            self.shallow_convs.append(
                nn.Sequential(
                    *[LargeKernelConv(
                        dims[i]
                    ) for _ in range(depths[i])]
                )
            )
        # -------------------------
        # GSC + Mamba (stage 2-3)
        # -------------------------
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(4):
            if i < 2:
                self.gscs.append(nn.Identity())
                self.stages.append(nn.Identity())
            else:
                self.gscs.append(
                    nn.Sequential(*[GSC(dims[i]) for _ in range(depths[i])])
                )

                if mamba_kwargs_per_stage and len(mamba_kwargs_per_stage) == 4:
                    kwargs = mamba_kwargs_per_stage[i] or {}
                else:
                    d_ssm_val = min(dims[i] // 2, 128 if i == 2 else 256)
                    kwargs = dict(
                        headdim=min(64, dims[i] // 4),
                        d_state=64,
                        d_ssm=d_ssm_val,
                        ngroups=1,
                    )

                self.stages.append(
                    nn.Sequential(
                        *[
                            HilbertSFCMambaLayer(
                                dim=dims[i],
                                mamba_cls=mamba_cls,
                                use_rope=self.use_rope,
                                **kwargs,
                            )
                            for _ in range(depths[i])
                        ]
                    )
                )

        # -------------------------
        # Norm per scale
        # -------------------------
        self.norms = nn.ModuleList(
            [nn.InstanceNorm3d(d, affine=False) for d in dims]
        )

       
        self.mlps = nn.ModuleList()

        for i, d in enumerate(dims):
            self.mlps.append(MlpChannel(d, 2 * d))
            

    def forward(self, x: torch.Tensor):
        outs = []

        for i in range(4):
            x = self.downsample_layers[i](x)

            if i < 2:
                x = self.fgms[i](x)
                x = self.shallow_convs[i](x)
            else:
                x = self.fgms[i](x)
                x = self.gscs[i](x)
                x = self.stages[i](x)

            if i in self.out_indices:
                x_norm = self.norms[i](x)

                xo = self.mlps[i](x_norm)

                outs.append(xo)

        return tuple(outs)



# ===================== FHSFMamba 主体 =====================

class FHSFMamba(nn. Module):
    
    
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        hidden_size: int = 768,
        norm_name="instance",
        res_block: bool = True,
        spatial_dims=3,
        mamba_cls=None,  # Mamba2
        mamba_kwargs_per_stage: Optional[List] = None,
        use_fgm: bool = True,
        use_rope:  bool = True,
        use_kan: bool = False,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.use_fgm = use_fgm
        self.use_rope = use_rope
        self.deep_supervision = deep_supervision
     

        # ============ 编码器第一层 ============
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # ============ 编码器骨干 ============
        self. vit = MambaEncoder(
            in_chans=feat_size[0],
            depths=depths,
            dims=feat_size,
            mamba_cls=mamba_cls,
            mamba_kwargs_per_stage=mamba_kwargs_per_stage,
            use_fgm=use_fgm,
            use_rope=use_rope,
            use_kan=use_kan,
        )

        # ============ 跳跃连接对齐 ============
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        # ============ 解码器 ============

        self.decoder5 = GDySampleUpBlock3D(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            #upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = GDySampleUpBlock3D(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            #upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = GDySampleUpBlock3D(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            #upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = GDySampleUpBlock3D(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            #upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=self.out_chans
        )

        if self.deep_supervision:
            self.ds_out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[3], out_channels=out_chans)
            self.ds_out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[2], out_channels=out_chans)
            self.ds_out1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=out_chans)

    def forward(self, x_in, voxel_spacing: Optional[Sequence[float]] = None):
        """
        前向传播
        
        Args:
            x_in: [B, C, D, H, W] 输入图像
        
        Returns:
            [B, num_classes, D, H, W] 分割结果
        """
        # ============ 编码器 ============
        enc1 = self.encoder1(x_in)  # [B, 48, D, H, W]
        z0, z1, z2, z3 = self.vit(enc1)  # 多尺度特征
        
        # ============ 跳跃连接对齐 ============
        s64 = self.encoder2(z0)  # [B, 96, D/2, H/2, W/2]
        s32 = self.encoder3(z1)  # [B, 192, D/4, H/4, W/4]
        s16 = self.encoder4(z2)  # [B, 384, D/8, H/8, W/8]
        enc5_input = self.encoder5(z3)  # [B, 768, D/16, H/16, W/16]

        # ============ 解码器 ============
        dec3 = self.decoder5(enc5_input, s16, voxel_spacing=voxel_spacing)
        dec2 = self.decoder4(dec3, s32, voxel_spacing=voxel_spacing)
        dec1 = self.decoder3(dec2, s64, voxel_spacing=voxel_spacing)
        dec0 = self.decoder2(dec1, enc1, voxel_spacing=voxel_spacing)
        out = self.decoder1(dec0)

        logits = self.out(out)

        if self.deep_supervision and self.training:
            out3 = self.ds_out3(dec3)
            out2 = self.ds_out2(dec2)
            out1 = self.ds_out1(dec1)
            return [logits, out3, out2, out1]

        return logits


  