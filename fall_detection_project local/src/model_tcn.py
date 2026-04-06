import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)

class PoseTCN(nn.Module):
    def __init__(self, in_dim=99, channels=128, layers=4, dropout=0.2):
        super().__init__()
        blocks = []
        ch = in_dim
        for i in range(layers):
            blocks.append(TCNBlock(ch, channels, k=3, dilation=2**i, dropout=dropout))
            ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        # x: (B,T,D) -> (B,D,T)
        x = x.transpose(1, 2)
        z = self.tcn(x)
        return self.head(z)  # (B,1)
