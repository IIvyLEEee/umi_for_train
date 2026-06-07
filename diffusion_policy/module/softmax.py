import torch
import numpy as np
import scipy
from torch.nn.functional import softmax

"""
Layer:  Softmax
Author: cxz21
Data:   2024/10/10
"""
def bin2fp16(fp16_bin):
    # Parse the binary string
    sign = int(fp16_bin[0])  # Sign bit
    exponent = int(fp16_bin[1:6], 2)  # Exponent bits
    fraction = int(fp16_bin[6:], 2)  # Fraction bits

    # Calculate the float value
    if exponent == 0 and fraction == 0:
        # Special case: zero
        return 0.0

    elif exponent == 0x1F:
        # Special case: infinity or NaN, based on fraction bits
        if fraction == 0:
            return float("inf") if sign == 0 else float("-inf")
        else:
            return float("nan")

    else:
        # Normal case: calculate float value
        # Bias of fp16 exponent is 15
        exponent_value = exponent - 15
        fraction_value = fraction / (
            2**10
        )  # Divide by 2^10 because fp16 has 10 fraction bits

        float_value = (-1) ** sign * (1 + fraction_value) * 2**exponent_value
        return float_value

class SoftmaxFunc(torch.autograd.Function):
    """
    Custom autograd function for Softmax in BF16 precision.
    """

    @staticmethod
    def forward(ctx, input: torch.tensor, k: torch.Tensor, b: torch.Tensor, array: torch.Tensor):
        input = input.to(torch.float16)
        x_max = torch.max(input, dim=-1, keepdim=True)[0]
        x_exp = input - x_max
        x_exp_flat = x_exp.view(-1)
        x_exp_result = x_exp_flat.clone()
        for m in range(15):
            mask = (x_exp_flat <= array[m]) & (x_exp_flat > array[m + 1])
            x_exp_result[mask] = x_exp_flat[mask] * k[m] + b[m]
        x_exp_result = x_exp_result.view(x_exp.size())
        output = x_exp_result / torch.sum(x_exp_result, dim=-1, keepdim=True)
        # print(output[0][0][:, :])
        # exit()
        # x_exp = torch.exp(input - x_max)
        # output = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
        # for backward
        ctx.save_for_backward(input, output)

        return output.to(torch.float16)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        input, output = ctx.saved_tensors
        input_grad = output * output_grad
        sum_input_grad = input_grad.sum(dim=-1, keepdim=True)
        input_grad -= output * sum_input_grad
        return input_grad


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = []
        self.b = []
        self.array = []
        x0s = [
            "1000000000000000",  # 0
            "1001100000000000",  # 1
            "1001110000000000",  # 2
            "1010000000000000",  # 3
            "1010010000000000",  # 4
            "1010100000000000",  # 5
            "1010110000000000",  # 6
            "1011000000000000",  # 7
            "1011010000000000",  # 8
            "1011100000000000",  # 9
            "1011110000000000",  # 10
            "1100000000000000",  # 11
            "1100010000000000",  # 12
            "1100100000000000",  # 13
            "1100110000000000",  # 14
            "1101000000000000",  # 15
            "1111110000000000",  # 16
        ]
        for i in range(len(x0s)):
            self.array.append(bin2fp16(x0s[i]))

        for i in range(len(self.array) - 2):
            x = np.linspace(self.array[i], self.array[i + 1], 1000)
            y = np.exp(x)
            # print(y)
            def func(x, k, b):
                return k * x + b
            popt, _ = scipy.optimize.curve_fit(func, x, y)
            self.k.append(popt[0])
            self.b.append(popt[1])

        self.k.append(0)
        self.b.append(0)

        self.array = torch.tensor(self.array, dtype=torch.float16)
        self.k = torch.tensor(self.k, dtype=torch.float16)
        self.b = torch.tensor(self.b, dtype=torch.float16)

    def forward(self, input:torch.Tensor):
        # return SoftmaxFunc.apply(input, self.k, self.b, self.array)
        input = input.to(torch.float16)
        self.array = self.array.to(input.device)
        self.k = self.k.to(input.device)
        self.b = self.b.to(input.device)

        x_max = torch.max(input, dim=-1, keepdim=True)[0]

        x_exp = input - x_max
        x_exp_flat = x_exp.view(-1)
        x_exp_result = x_exp_flat.clone()
        for m in range(15):
            mask = (x_exp_flat <= self.array[m]) & (x_exp_flat > self.array[m + 1])
            x_exp_result[mask] = x_exp_flat[mask] * self.k[m] + self.b[m]
        x_exp_result = x_exp_result.view(x_exp.size())

        recipe = 1 / torch.sum(x_exp_result, dim=-1, keepdim=True)

        output = x_exp_result * recipe

        return output.to(torch.float16)


if __name__ == "__main__":
    batch_size = 2
    X = 5
    Y = 3
    input = torch.randn(batch_size, X, Y, requires_grad=True).to("cuda")
    x = input.detach().clone().requires_grad_(True)
    y = input.detach().clone().requires_grad_(True)
    layer = Softmax()
    out = layer(x)
    outy = softmax(y, dim=-1)

    print("out: ", out)
    print("outy: ", outy)

    dout = torch.randn(batch_size, X, Y).to("cuda")

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    print("dx: ", x.grad)
    print("dy: ", y.grad)
