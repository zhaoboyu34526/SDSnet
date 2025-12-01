import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import collections.abc
from itertools import repeat
import math
import warnings
from torch import Tensor
from DCNv4 import DCNv4

try:
    import selective_scan_cuda_core
except Exception as e:
    print(f"WARNING: can not import selective_scan_cuda_core.")


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def antidiagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        return y_res


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan):


    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    return ys

class OSSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        self.norm1 = nn.LayerNorm(d_inner)
        self.norm2 = nn.LayerNorm(d_inner)

        k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        self.act = nn.SiLU()

        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv) // 2,
        )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x, hw_shape):

        x = self.in_proj(x)
        x, z = x.chunk(2, dim=2)
        z = self.act(z)

        x = nlc_to_nchw(x, hw_shape)
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        ys = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            force_fp32=True,
            SelectiveScan=SelectiveScanCore,
        )

        B, K, D, H, W = ys.shape
        ys = ys.view(B, K, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        y_res = nchw_to_nlc(y_res)

        y = z * y_res
        out = self.dropout(self.out_proj(y))
        return out


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float,
                           b: float) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=None,
                 padding=1,
                 dilation=1,
                 bias=True):
        super().__init__()

        self.embed_dims = embed_dims

        if stride is None:
            stride = kernel_size

        self.projection = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=embed_dims,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)

        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MixFFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        in_channels = embed_dims

        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class DeformMixFFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        in_channels = embed_dims

        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1)

        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1)

        self.dcnv4 = DCNv4(
                    channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=(3 - 1) // 2,
                    group=2)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        x = x + self.dropout_layer(self.norm1(self.dcnv4(x)))
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        out = x + self.dropout_layer(self.norm2(out))
        return out


class DeformMambaEncoderLayer(nn.Module):
    """
    Implements one encoder layer compose of mamba and DeformMixFFN in UVMamba.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class DeformMixVisionMamba(nn.Module):
    """The backbone of uvmamba.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                DeformMambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# Ablation 1：without SADE
# ====================================

class MambaEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class MixVisionMamba(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                MambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

# ====================================
# Ablation 2: without MSSM
# ====================================

class DeformEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class DeformMixVision(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                DeformEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# Ablation 3：SADE and MSSM are arranged parallel
# ====================================
class ParallelDeformMambaEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x_deform = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x_ssm = self.mix_ffn(x, hw_shape, identity=x)
        x = self.norm2(x_deform + x_ssm)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class ParallelDeformMixVisionMamba(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                ParallelDeformMambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# Ablation 4 : MSSM --> SADE
# ====================================

class MambaDeformEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]

        x = self.norm1(x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class MambaMixVisionDeform(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                MambaDeformEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

class Decoder(nn.Module):
    def __init__(self, inchannels, num_classes, channels=256, interpolate_mode='bilinear', dropout_ratio=0.1):
        super().__init__()

        self.in_channels = inchannels
        self.channels = channels
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.up1 = nn.ConvTranspose2d(256, 160, 2, stride=2)
        self.conv1 = DoubleConv(320, 160)

        self.up2 = nn.ConvTranspose2d(160, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = DoubleConv(64, 32)

        self.up4 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.conv4 = DoubleConv(64, 64)

        self.cls_seg = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = inputs[::-1]
        x1, x2, x3, x4 = inputs

        up1 = self.up1(x1)
        merge1 = torch.concat([up1, x2], dim=1)
        conv1 = self.conv1(merge1)

        up2 = self.up2(conv1)
        merge2 = torch.concat([up2, x3], dim=1)
        conv2 = self.conv2(merge2)

        up3 = self.up3(conv2)
        merge3 = torch.concat([up3, x4], dim=1)
        conv3 = self.conv3(merge3)

        up4 = self.up4(conv3)
        conv4 = self.conv4(up4)

        out = self.cls_seg(conv4)
        return out


class UVMamba(nn.Module):
    def __init__(self, config):
        super(UVMamba, self).__init__()

        self.backbone = DeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



# ====================================
# Ablation 1: without SADE
# ====================================

class UVMambaNoDeform(nn.Module):
    def __init__(self, config):
        super(UVMambaNoDeform, self).__init__()

        self.backbone = MixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


# ====================================
# Ablation 2: without MSSM
# ====================================

class UVMambaNoSSM(nn.Module):
    def __init__(self, config):
        super(UVMambaNoSSM, self).__init__()

        self.backbone = DeformMixVision(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ==================================================
# Ablation 3: SADE and MSSM are arranged parallel
# ==================================================


class UVMambaParallel(nn.Module):
    def __init__(self, config):
        super(UVMambaParallel, self).__init__()

        self.backbone = ParallelDeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ====================================
# Ablation 4: SSM --> DCN
# ====================================


class UVMambaReverse(nn.Module):
    def __init__(self, config):
        super(UVMambaReverse, self).__init__()

        self.backbone = MambaMixVisionDeform(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
