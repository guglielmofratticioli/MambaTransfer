import torch
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

batch, length, dim = 1, 64, 32
x = torch.randn(batch, length, dim).to("cuda")

 # assert self.d_inner % self.headdim == 0
 # self.d_inner = self.expand * self.d_model

 # causal conv 1d  stride rule -> -> d_model * expand / headdim = multiple of 8
model = Mamba2Simple(
    d_model=dim,  # Model dimension d_model
    headdim = 8,
    d_state = 64,
    use_mem_eff_path = False
).to("cuda")
y = model(x)
assert y.shape == x.shape