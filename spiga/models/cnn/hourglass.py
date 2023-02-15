import torch.nn as nn

from spiga.models.cnn.layers import Conv, Deconv, Residual


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Conv(f, f, 2, 2,  bn=True, relu=True)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = Deconv(f, f, 2, 2, bn=True, relu=True)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassCore(Hourglass):
    def __init__(self, n, f, bn=None, increase=0):
        super(HourglassCore, self).__init__(n, f, bn=bn, increase=increase)
        nf = f + increase
        if self.n > 1:
            self.low2 = HourglassCore(n - 1, nf, bn=bn)

    def forward(self, x, core=[]):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        if self.n > 1:
            low2, core = self.low2(low1, core=core)
        else:
            low2 = self.low2(low1)
            core.append(low2)
        low3 = self.low3(low2)
        if self.n > 1:
            core.append(low3)
        up2 = self.up2(low3)
        return up1 + up2, core
