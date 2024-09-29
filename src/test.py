import torch
import timm
from unet_model import VovUnet_Var
from timm.models.eva import EvaBlock,EvaAttention
from timm.models.vovnet import OsaBlock
# a_tensor = torch.randn(16, 16, 16)# Batch, N

# test_model = EvaAttention(16)
# out = test_model(a_tensor)
# print(out.shape)

# a_tensor = torch.randn(16, 16, 16)# Batch, N

# test_model = EvaBlock(dim=16, num_heads=4)
# out = test_model(a_tensor)
# print(out.shape)

# a_tensor = torch.randn(16, 16, 24, 24)# Batch, N, H, W

# test_model = OsaBlock(in_chs=16, mid_chs=16, out_chs=32,layer_per_block=2)
# out = test_model(a_tensor)
# print(out.shape)

# a_tensor = torch.randn(16, 1, 256, 256)# Batch, N, H, W

# test_model = VovUnet_Var(1, 1, 11)
# out = test_model(a_tensor)
# print(out[0].shape)
# print(out[1].shape)

# # params
# print(sum(p.numel() for p in test_model.parameters() if p.requires_grad)/1e6)

print(timm.list_models('u*'))