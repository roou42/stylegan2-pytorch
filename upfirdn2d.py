# upfirdn2d.py (Python fallback)
import torch
import torch.nn.functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    input = input.reshape(-1, 1, in_h, in_w)

    # Upsample
    if up > 1:
        input = F.pad(input, (0, 0, 0, 0))
        input = input.reshape(-1, 1, in_h, 1, in_w, 1)
        input = F.pad(input, (0, 0, 0, 0, 0, 0))
        input = input.expand(-1, -1, -1, up, -1, up)
        input = input.reshape(-1, 1, in_h * up, in_w * up)

    # Padding
    input = F.pad(input, (pad[0], pad[1], pad[0], pad[1]))

    # Convolution with flipped kernel
    kernel = kernel.to(input.device, dtype=input.dtype)
    kernel = kernel.flip([0, 1]).unsqueeze(0).unsqueeze(0)
    input = F.conv2d(input, kernel, stride=1, groups=input.shape[1])

    # Downsample
    if down > 1:
        input = input[:, :, ::down, ::down]

    input = input.reshape(batch, channel, input.shape[2], input.shape[3])
    return input
