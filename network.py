import torch.nn as  nn
import torch
import torch.nn.functional as F
import math
def Conv1d_with_init(in_channels, out_channels, kernel_size):
   layer = nn.Conv1d(in_channels, out_channels, kernel_size)
   nn.init.kaiming_normal_(layer.weight)
   return layer


class Encoder(nn.Module):
   def __init__(self, seq_len, enc_in):
      super().__init__()
      self.norm1 = nn.LayerNorm(seq_len)
      self.ff1 = nn.Sequential(
         nn.Linear(seq_len, seq_len),
         nn.GELU(),
         nn.Dropout(0.1)
      )
      self.ff2 = nn.Sequential(
         nn.Linear(enc_in, enc_in),
         nn.GELU(),
         nn.Dropout(0.1)
      )
   def forward(self, x):
      B ,C ,T = x.shape    #B,C,L
      y_0 = self.ff1(x)
      y_0 = y_0 + x
      y_0 = self.norm1(y_0)
      y_1 = y_0.permute(0, 2, 1)
      y_1 = self.ff2(y_1)
      y_1 = y_1.permute(0, 2, 1)
      y_2 = y_1 * y_0 + x
      y_2 = self.norm1(y_2)
      return y_2


class SelfAttentionEncoder(nn.Module):
   def __init__(self, input_dim, embed_dim=128, num_heads=8, dropout=0.1):
      super().__init__()
      self.proj_in = nn.Linear(input_dim, embed_dim)
      self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
      self.ffn = nn.Sequential(
         nn.Linear(embed_dim, embed_dim * 4),
         nn.GELU(),
         nn.Dropout(dropout),
         nn.Linear(embed_dim * 4, embed_dim)
      )
      self.norm1 = nn.LayerNorm(embed_dim)
      self.norm2 = nn.LayerNorm(embed_dim)
      self.proj_out = nn.Linear(embed_dim, input_dim)

   def forward(self, x):
      """
      x: [B, N, T]
      return: [B, N, T]
      """
      B, N, T = x.shape

      x = x.transpose(1, 2)


      h = self.proj_in(x)  # [B, T, d_model]

      # Self-Attention
      attn_out, _ = self.attn(h, h, h)
      h = self.norm1(h + attn_out)

      # Feed Forward
      ffn_out = self.ffn(h)
      h = self.norm2(h + ffn_out)


      out = self.proj_out(h)  # [B, T, c_in]


      out = out.transpose(1, 2)
      return out

class DiffusionEmbedding(nn.Module):
   def __init__(self, dim, proj_dim, max_steps=500):
      super().__init__()
      self.register_buffer(
         "embedding", self._build_embedding(dim, max_steps), persistent=False
      )
      self.projection1 = nn.Linear(dim * 2, proj_dim)
      self.projection2 = nn.Linear(proj_dim, proj_dim)

   def forward(self, diffusion_step):
      x = self.embedding[diffusion_step]
      x = self.projection1(x)
      x = F.silu(x)
      x = self.projection2(x)
      x = F.silu(x)
      return x  # [batch_size, proj_dim]

   def _build_embedding(self, dim, max_steps):
      steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
      dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
      table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
      table = torch.cat(
         [torch.sin(table), torch.cos(table)], dim=1)  # [T,2*dim]
      return table
class TCNBlock(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
      super(TCNBlock, self).__init__()

      padding = (kernel_size - 1) * dilation

      self.conv1 = nn.Conv1d(
         in_channels=in_channels,
         out_channels=out_channels,
         kernel_size=kernel_size,
         padding=padding,
         dilation=dilation,
         bias=False
      )
      self.layer_norm1 = nn.LayerNorm(out_channels)
      self.relu1 = nn.ReLU()
      self.dropout1 = nn.Dropout(dropout)

      self.conv2 = nn.Conv1d(
         in_channels=out_channels,
         out_channels=out_channels,
         kernel_size=kernel_size,
         padding=padding,
         dilation=dilation,
         bias=False
      )
      self.layer_norm2 = nn.LayerNorm(out_channels)
      self.relu2 = nn.ReLU()
      self.dropout2 = nn.Dropout(dropout)


      self.residual_conv = None
      if in_channels != out_channels:
         self.residual_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
         )

   def forward(self, x):
      input_seq_len = x.shape[2]
      residual = x

      out = self.conv1(x)
      out = out[:, :, :input_seq_len]
      out = out.transpose(1, 2)
      out = self.layer_norm1(out)
      out = out.transpose(1, 2)
      out = self.relu1(out)
      out = self.dropout1(out)
      out = self.conv2(out)
      out = out[:, :, :input_seq_len]
      out = out.transpose(1, 2)
      out = self.layer_norm2(out)
      out = out.transpose(1, 2)
      out = self.relu2(out)
      out = self.dropout2(out)
      if self.residual_conv is not None:
         residual = self.residual_conv(residual)
      assert out.shape == residual.shape, \
         f"Shape mismatch in residual connection! out: {out.shape}, residual: {residual.shape}"
      return out + residual

class TCN(nn.Module):
   def __init__(self, input_dim, num_channels=[64, 32], kernel_size=3, dropout=0.2,d=2):
      super(TCN, self).__init__()
      self.input_dim = input_dim
      in_channels = input_dim
      layers = []
      dilation_size = d
      for out_channels in num_channels:
         layers.append(TCNBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_size,
            dropout=dropout
         ))
         in_channels = out_channels
         dilation_size *= 2
      self.tcn = nn.Sequential(*layers)

   def forward(self, x):
      tcn_out = self.tcn(x)
      return tcn_out


class ResidualBlock(nn.Module):
   def __init__(self, residual_channels,hidden_size, dilation,seqlenth,dilation_exp=None):
      super().__init__()
      self.dilated_conv = nn.Conv1d(
         residual_channels,
         residual_channels*2,
         3,
         padding=1,
         dilation=1,
      )
      self.dropout=0.05
      self.length=seqlenth
      self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
      self.Econder =Encoder(seq_len= self.length, enc_in=residual_channels)
      self.SelfAttentionEncoder=SelfAttentionEncoder(input_dim=residual_channels)
      self.TCN_1 = TCN(residual_channels, num_channels=[128, residual_channels], kernel_size=3, dropout=0.05, d=2)
      self.TCN_2 = TCN(residual_channels, num_channels=[128, residual_channels], kernel_size=3, dropout=0.05, d=3)
      self.output_projection = nn.Conv1d(
         int(residual_channels),residual_channels*2, 1)
      self.conv1d = nn.Conv1d(
          residual_channels, residual_channels*2, 1)

      self.dropout_layer = nn.Dropout(self.dropout)

      nn.init.kaiming_normal_(self.output_projection.weight)
      nn.init.kaiming_normal_(self.conv1d.weight)

   def forward(self, x, conditioner,diffusion_step):
      diffusion_step = self.diffusion_projection(
         diffusion_step).unsqueeze(-1)

      x_conditioner =self.Econder(conditioner)

      x_conditioner1 =self.TCN_1(x_conditioner)
      x_conditioner2 = self.TCN_2(x_conditioner)

      x_conditioner =torch.cat([x_conditioner1,x_conditioner2],dim=1)


      y = x + diffusion_step

      y = self.dilated_conv(y)+x_conditioner
      gate, filter = torch.chunk(y, 2, dim=1)
      y = torch.sigmoid(gate) * torch.tanh(filter)

      y = self.output_projection(y)
      y = F.leaky_relu(y, 0.4)
      #y =F.relu(y)
      residual, skip = torch.chunk(y, 2, dim=1)
      return (x + residual) / math.sqrt(2.0), skip

class diff_CSDI(nn.Module):
   def __init__(self,config):
      super().__init__()
      self.channels = config["diffusion"]["channels"]
      self.num_features = config["others"]["feature_num"]
      self.seqlenth =config["others"]["eval_length"]
      self.layer = config["model"]["layers"]
      self.d_model = config["model"]["d_model"]
      self.nhead = 4
      self.num_layers = 4
      self.dropout = 0.05

      self.input_projection = nn.Conv1d(
         self.num_features, self.d_model, 1
      )  # 1D convolution shape [batch_size, residual_channels, target_dim]
      self.diffusion_embedding = DiffusionEmbedding(
         16, proj_dim=self.d_model
      )  # Time embedding shape [batch_size, proj_dim]
      self.residual_layers = nn.ModuleList(
         [
            ResidualBlock(
               residual_channels=self.d_model,
               hidden_size=self.d_model,
               dilation=2 ** (i % 10),
               seqlenth =self.seqlenth
            )
            for i in range(self.layer)
         ]
      )
      self.input_projection_noise = nn.Conv1d(self.num_features, self.d_model, 1)
      self.skip_projection = nn.Conv1d(
         self.d_model, self.d_model, 1)  # Skip connection
      self.output_projection = Conv1d_with_init(self.d_model, 4, 1)
      self.linear = nn.Linear(self.d_model,self.num_features)
      self.flatten = nn.Flatten(start_dim=-2)
      nn.init.kaiming_normal_(self.input_projection.weight)
      nn.init.kaiming_normal_(self.skip_projection.weight)
      nn.init.kaiming_normal_(self.input_projection_noise.weight)
      nn.init.zeros_(self.output_projection.weight)
   def forward(self,  xT, x0, conda_mask, time):
      B, K, L = xT.shape

      x_noise = self.input_projection_noise(xT)   #[B,32,L]
      x =F.relu(x_noise)
      diffusion_step = self.diffusion_embedding(time)

      cond_up = x0 * conda_mask

      cond_up = self.input_projection(cond_up)  # [B,32,L]
      cond_up=F.relu(cond_up)

      skip = []
      for layer in self.residual_layers:
         x, skip_connection = layer(x, cond_up, diffusion_step)
         skip.append(skip_connection)

      x = torch.sum(torch.stack(skip), dim=0) / \
          math.sqrt(len(self.residual_layers))  # [B,8,T]


      x = self.skip_projection(x)  # [B,8,T]
      x = F.relu(x).transpose(1,2)

      x =self.linear(x).transpose(1,2)


      return x

def prediction_fusion(preds, weights=[5,3,2]):
     # weights = [6, 2, 1, 1]
      """


      Args:
          preds:  (B, S, L, F)
          weights:

      Returns:
          fused: (B, S, L, F)
      """
      num_scales = int(len(preds))

      assert len(preds) == num_scales, f"The prediction results at input{num_scales} granularities."
      for p in preds:
         assert p.shape == preds[0].shape, "The shapes at all granularities must be consistent (B, S, L, F)"

      B, S, L, F = preds[0].shape
      weights_tensor = torch.tensor(weights, dtype=torch.float32, device=preds[0].device)
      weights_normalized = torch.softmax(weights_tensor, dim=0)
      fused = torch.zeros_like(preds[0])
      for i in range(num_scales):
         fused += weights_normalized[i] * preds[i]

      return fused