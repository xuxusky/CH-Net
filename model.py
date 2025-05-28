import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from modules.transformer import TransformerEncoder
import math


class PHMLinear(nn.Module):

    def __init__(self, n, in_features, out_features, cuda=True, gate=True):
        super(PHMLinear, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda
        self.gate_enabled = gate

        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

        self.S = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features // n, self.in_features // n))))

        self.weight = torch.zeros((self.out_features, self.in_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

        # Optional gating network (output-based)
        if self.gate_enabled:
            self.gate_net = nn.Sequential(
                nn.Linear(self.out_features, self.out_features),
                nn.Sigmoid()
            )

    def kronecker_product1(self, a, b):  # adapted from Bayer Research's implementation
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features))
        for i in range(self.n):
            H = H + torch.kron(self.A[i], self.S[i])
        return H

    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        out = F.linear(input.type_as(self.weight), weight=self.weight, bias=self.bias)
        # Apply gating if enabled
        if self.gate_enabled:
            gate_vals = self.gate_net(out)
            out = out * gate_vals
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, gate={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.gate_enabled)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.S, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class eyeBase(nn.Module):
    "Base for the eye Modality."

    def __init__(self, units=1024):
        super(eyeBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(600 * 1, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x


class GSRBase(nn.Module):
    "Base for the GSR Modality."

    def __init__(self, units=1024):
        super(GSRBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280 * 1, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x


class EEGBase(nn.Module):
    "Base for the EEG Modality."

    def __init__(self, units=1024):
        super(EEGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280 * 1, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x


class ECGBase(nn.Module):
    "Base for the ECG Modality."

    def __init__(self, units=1024):
        super(ECGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280 * 1, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x


class HyperFuseNet(nn.Module):
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""

    def __init__(self, dropout_rate=0.2, units=1024, n=4, class_num=3):
        super(HyperFuseNet, self).__init__()  # call the parent constructor
        self.eye = eyeBase()
        self.gsr = GSRBase()
        self.eeg = EEGBase()
        self.ecg = ECGBase()
        kernel_size = 1
        # 1. Temporal convolutional layers
        self.proj_m1 = nn.Conv1d(4, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.proj_m2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.proj_m3 = nn.Conv1d(10, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.proj_m4 = nn.Conv1d(3, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.final_conv = nn.Conv1d(4, 1, kernel_size=kernel_size, padding=0, bias=False)

        self.trans_m1_all = TransformerEncoder(units, 4, 2, attn_dropout=0.05, relu_dropout=0.1, res_dropout=0.1,
                                               embed_dropout=0.1, attn_mask=False)
        self.trans_m2_all = TransformerEncoder(units, 4, 2, attn_dropout=0.05, relu_dropout=0.1, res_dropout=0.1,
                                               embed_dropout=0.1, attn_mask=False)
        self.trans_m3_all = TransformerEncoder(units, 4, 2, attn_dropout=0.05, relu_dropout=0.1, res_dropout=0.1,
                                               embed_dropout=0.1, attn_mask=False)
        self.trans_m4_all = TransformerEncoder(units, 4, 2, attn_dropout=0.05, relu_dropout=0.1, res_dropout=0.1,
                                               embed_dropout=0.1, attn_mask=False)


        self.flat = nn.Flatten()
        self.drop = nn.Dropout(dropout_rate)
        self.D1 = PHMLinear(n, units * 4, units * 2)

        # self.fc1 = nn.Linear(units*4, units*2)
        self.BN1 = nn.BatchNorm1d(units * 2)
        self.drop1 = nn.Dropout(dropout_rate)

        self.D2 = PHMLinear(n, units * 2, units)

        # self.fc2 = nn.Linear(units * 2, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.drop2 = nn.Dropout(dropout_rate)

        self.D3 = PHMLinear(n, units, units // 2)
        # self.BN3 = nn.BatchNorm1d(units//2)
        # self.fc3 = nn.Linear(units, units//2)
        self.BN3 = nn.BatchNorm1d(units // 2)
        self.drop3 = nn.Dropout(dropout_rate)

        self.D4 = PHMLinear(n, units // 2, units // 4)
        # self.fc4 = nn.Linear(units//2, units // 4)
        self.drop4 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(units // 4, class_num)

    def forward(self, eye, gsr, eeg, ecg):
        gsr = torch.unsqueeze(gsr, 2)

        eye = eye.transpose(1, 2)
        gsr = gsr.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        ecg = ecg.transpose(1, 2)

        eye_out = self.proj_m1(eye)
        gsr_out = self.proj_m2(gsr)
        eeg_out = self.proj_m3(eeg)
        ecg_out = self.proj_m4(ecg)  # (32,1,1280)

        eye_out = self.eye(eye_out)  # (32,1024)
        gsr_out = self.gsr(gsr_out)  # (32,1024)
        eeg_out = self.eeg(eeg_out)  # (32,1024)
        ecg_out = self.ecg(ecg_out)  # (32,1024)

        eye_out = torch.unsqueeze(eye_out, 0)  # (1, 32,1024)
        gsr_out = torch.unsqueeze(gsr_out, 0)  # (1, 32,1024)
        eeg_out = torch.unsqueeze(eeg_out, 0)  # (1, 32,1024)
        ecg_out = torch.unsqueeze(ecg_out, 0)  # (1, 32,1024)
        proj_all = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=0)  # (4, 32, 1024)

        # concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)  # (32,4096)
        eye_with_all = self.trans_m1_all(eye_out, proj_all, proj_all)
        gsr_with_all = self.trans_m2_all(gsr_out, proj_all, proj_all)
        eeg_with_all = self.trans_m3_all(eeg_out, proj_all, proj_all)
        ecg_with_all = self.trans_m4_all(ecg_out, proj_all, proj_all)

        concat = torch.cat([eye_with_all, gsr_with_all, eeg_with_all, ecg_with_all], dim=0)
        concat = concat.permute(1, 0, 2)
        concat = self.flat(concat)  # (32,4096)

        x = self.D1(concat)
        # x = self.fc1(concat)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        # x = self.fc2(x)
        x = F.relu(self.BN2(x))
        x = self.drop(x)
        x = self.D3(x)
        # x = self.fc3(x)
        x = F.relu(self.BN3(x))
        # x = F.relu(self.fc4(x))
        x = F.relu(self.D4(x))
        out = self.out(x)  # Softmax would be applied directly by CrossEntropyLoss, because labels=classes
        return out


if __name__ == '__main__':
    eye = torch.randn(32, 600, 4).cuda()
    gsr = torch.randn(32, 1280).cuda()
    eeg = torch.randn(32, 1280, 10).cuda()
    ecg = torch.randn(32, 1280, 3).cuda()
    net = HyperFuseNet().cuda()
    outputs = net(eye, gsr, eeg, ecg)
    print(outputs.shape)

    from thop import profile
    import time
    import numpy as np

    flops, params = profile(net, (eye, gsr, eeg, ecg))
    print('flops:', flops, 'params:', params)
    print('flops: %.2f M, Gflops: %.2f G, params: %.2f M' % (
        flops / 1000000.0, flops / 1000000.0 / 1024, params / 1000000.0))

    times = []
    for i in range(20):
        eye = torch.randn(2, 600, 4).cuda()
        gsr = torch.randn(2, 1280).cuda()
        eeg = torch.randn(2, 1280, 10).cuda()
        ecg = torch.randn(2, 1280, 3).cuda()
        net = HyperFuseNet().cuda()
        start = time.time()
        outputs = net(eye, gsr, eeg, ecg)
        end = time.time()

        times.append(end - start)

    print(f"FPS: {2.0 / np.mean(times):.3f}")
