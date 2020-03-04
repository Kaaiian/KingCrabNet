import numpy as np

import torch
from torch import nn


# %%
class AttentionBlock(nn.Module):
    """
    This implements the multi-headed attention block
    of the CrabNet architecture.
    Parameters
    ----------
    d_model: int
        the number of expected features in the input (required, default=32).
    nhead: int
        the number of heads in the multiheadattention models (required,
        default=2).
    dim_feedforward: int
        the dimension of the feedforward network model (required, default=16).
    dropout: float
        the dropout value (default=0.1).
    edm: bool
        specifies whether the input X matrix is of type EDM
        or not (optional, default=False).
    """
    def __init__(self,
                 compute_device,
                 d_model=32,
                 nhead=2,
                 dim_feedforward=16,
                 dropout=0.1,
                 edm=False):
        super(AttentionBlock, self).__init__()

        self.compute_device = compute_device

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.edm = edm
        self.dropout = dropout

        self.softmax = nn.Softmax(dim=-1)

        self.layernorm0 = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.layernorm1a = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.layernorm1b = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.layernorm2a = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.layernorm2b = nn.LayerNorm(self.d_model, elementwise_affine=True)

        self.dropout1 = nn.Dropout(p=self.dropout)

        # self.fc_q = nn.Linear(self.d_model, self.dim_feedforward, bias=False)
        # self.fc_k = nn.Linear(self.d_model, self.dim_feedforward, bias=False)
        # self.fc_v = nn.Linear(self.d_model, self.dim_feedforward, bias=False)

        self.fc_q_list = nn.ModuleList(
                [nn.Linear(self.d_model, self.dim_feedforward, bias=True)
                  for _ in range(self.nhead)]
            )
        self.fc_k_list = nn.ModuleList(
                [nn.Linear(self.d_model, self.dim_feedforward, bias=True)
                  for _ in range(self.nhead)]
            )
        self.fc_v_list = nn.ModuleList(
                [nn.Linear(self.d_model, self.dim_feedforward, bias=True)
                  for _ in range(self.nhead)]
            )

        self.fc_o = nn.Linear(self.nhead * self.dim_feedforward,
                             self.d_model,
                             bias=True)

        self.fc1 = nn.Linear(self.d_model, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward pass of the attention block.
        Parameters
        ----------
        x: torch.Tensor
            A representation of the chemical compounds in the shape
            (batch, n_compounds, n_elements, n_feats) in the case of EDM data,
            (batch, n_compounds, n_feats) in the case of non-EDM data.
        Returns
        -------
        r: nn.Variable shape (a, b)
            The result of the forward pass through the attention block.
        """
        sqrt = np.sqrt(self.dim_feedforward)
        sqrt = torch.as_tensor(sqrt).to(self.compute_device)

        x = self.layernorm0(x)

        # Self-attention
        z_list = self.nhead * [None]
        for i in range(self.nhead):
            q = self.fc_q_list[i](x)
            k = self.fc_k_list[i](x)
            v = self.fc_v_list[i](x)
            # q = self.leaky(self.fc_q_list[i](x))
            # k = self.leaky(self.fc_k_list[i](x))
            # v = self.leaky(self.fc_v_list[i](x))
            # q = self.fc_q(x)
            # k = self.fc_k(x)
            # v = self.fc_v(x)
            # q = self.leaky(self.fc_q(x))
            # k = self.leaky(self.fc_k(x))
            # v = self.leaky(self.fc_v(x))

            k_t = torch.transpose(k, dim0=-2, dim1=-1)
            qk_t = torch.matmul(q, k_t)

            soft = self.softmax(qk_t / sqrt)
            # eye = torch.eye(4).to(self.compute_device)
            # soft = soft + eye
            # soft.register_hook(lambda x: print(x[0]))
            soft = self.dropout1(soft)
            # if i == 0:
            #     print(soft[0, :, :])
            z = torch.matmul(soft, v)
            z_list[i] = z

        z = torch.cat(z_list, dim=-1)
        z = self.fc_o(z)

        # Feed-forward
        r0 = x + z
        r0 = self.layernorm1a(r0)
        r0 = self.fc1(r0)
        r0 = self.layernorm1b(r0)

        r1 = r0 + x
        r1 = self.layernorm2a(r1)
        r = self.fc2(r1)
        r = self.layernorm2b(r)

        return r


# %%
class CrabNet(nn.Module):
    """
    This implements the overall CrabNet architecture.
    Parameters
    ----------
    input_dims: int
        the number of expected features in the input (required).
    d_model: int
        the number of element embedding dimensions (optional, default=64).
    nhead: int
        the number of heads in the multi-headed attention mechanism (optional,
        default=4).
    num_layers: int
        the number of sub-encoder-layers in the encoder (optional, default=2).
    dim_feedforward: int
        the dimension of the feedforward network model (optional, default=16).
    dropout: float
        the dropout value (optional, default=0.1).
    edm: bool
        specifies whether the input X matrix is of type EDM
        or not (optional, default=False).
    """

    def __init__(self,
                 compute_device,
                 input_dims,
                 d_model=201,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=16,
                 dropout=0.1,
                 edm=False):
        super(CrabNet, self).__init__()

        self.compute_device = compute_device

        self.input_dims = input_dims
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.edm = edm
        self.output_dims = 1

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.prelu1 = nn.PReLU(num_parameters=8)

        self.fcmask = nn.Linear(self.input_dims, 1, bias=False)
        self.fc1 = nn.Linear(self.input_dims, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        # self.fc3 = nn.Linear(self.d_model, self.output_dims)

        self.attentionblocks = nn.ModuleList(
                [AttentionBlock(compute_device=self.compute_device,
                                d_model=self.d_model,
                                nhead=self.nhead,
                                dim_feedforward=self.dim_feedforward,
                                dropout=self.dropout,
                                edm=self.edm)
                 for _ in range(self.num_layers)]
            )

        # define an output neural network
        out_hidden = [1024, 512, 256, 256, 128]
        self.output_nn = ResidualNetwork(self.d_model, 1, out_hidden)
        self.leaky = nn.LeakyReLU()
    def forward(self, x):
        """
        Forward pass of the CrabNet model.
        Parameters
        ----------
        x: torch.Tensor
            A representation of the chemical compounds in the shape
            (n_compounds, n_elements, n_feats) in the case of EDM data,
            (n_compounds, n_feats) in the case of non-EDM data.
        Returns
        -------
        y: torch.Tensor
            The element property prediction with the shape 1.
        """
        x0 = self.fcmask(x)
        x0 = self.leaky(x0)
        # x = self.dropout1(x)
        # print(x[0, :, :])
        for i, block in enumerate(self.attentionblocks):
            x = block(x)
            # if i == 0 and self.edm:
            #     x = self.prelu1(x)

        x = self.output_nn(x)
        # print(x.shape)
        # print(x0.shape)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # print(x[0, :, :])
        x = x * x0
        # print(x[0, :, :])

        if self.edm:
            x = torch.sum(x, dim=-2)

        y = x

        return y


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super(ResidualNetwork, self).__init__()

        dims = [input_dim]+hidden_layer_dims

        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1])
                                  for i in range(len(dims)-1)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
        #                           for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1])
                                      else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        # for fc, bn, res_fc, act in zip(self.fcs, self.bns,
        #                                self.res_fcs, self.acts):
        #     fea = act(bn(fc(fea)))+res_fc(fea)
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)

        return self.fc_out(fea)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
