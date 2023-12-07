import torch
from torch import nn, flip


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.network(x)
        return y


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Mlp, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        y = self.network(x)
        y[:,-1] = torch.sigmoid(y[:,-1])
        # y = torch.round(y)
        return y


class Conv_Net(nn.Module):
    def __init__(self, n_channels, output_dim):
        super(Conv_Net, self).__init__()
        # hidden_chanels = 64
        self.features_3x3 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size= (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        # self.dropout = nn.Dropout2d(p=0.5)
        # self.features_2x2 = nn.Sequential(
        #     nn.Conv2d(hidden_chanels, n_channels, kernel_size = 2),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, 2)
        #     )
        # self.classifier = nn.Linear(n_channels * 12, output_dim)

    def forward(self, x):
        # x = transform(x)
        x = self.features_3x3(x)
        # x1 = self.dropout(x1)
        # x2 = self.features_2x2(x1)
        y = x.flatten(1)
        if hasattr(self, 'classifier'):
            y = self.classifier(y)
        y[:,-1] = torch.sigmoid(y[:,-1])
        return y


# class Attention(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Attention, self).__init__()
#         self.attention = nn.MultiheadAttention(input_size, 1)
#         self.linear = nn.Linear(input_size, hidden_size)
#         self.classifier = nn.Linear(35, output_size)

#     def forward(self, x):        #(batch, seq, feature)
#         x = x.permute(1, 0, 2)   #(seq, batch, feature)
#         out, _ = self.attention(x, x, x)
#         out = out.permute(1, 0, 2)
#         out = self.linear(out)
#         out = self.classifier(out.flatten(1))
#         out[:, -1] = torch.sigmoid(out[:, -1])
#         return out


class Attention(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, 1)
        # self.linear = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(35*3, output_size)

    def forward(self, x):        #(batch, seq, feature)
        x = x.permute(1, 0, 2)   #(seq, batch, feature)
        out, _ = self.attention(x, x, x)
        out = out.permute(1, 0, 2)
        out = self.classifier(out.flatten(1))
        out[:, -1] = torch.sigmoid(out[:, -1])
        return out


def get_label(energy, mean = None):
    label = energy.clone()
    if mean and mean < float('inf'):
        energy_mean = mean
    else:
        energy_mean = energy.mean()
    for i in range(energy.shape[0]):
        label[i] = energy[i] < energy_mean
    return label


def change_code(x):
    """x- torch.Tensor,
    shape: (batch_size, arch_code_len),
    dtype: torch.float32"""
    pos_dict = {'00': 3, '01': 4, '10': 5, '11': 6}
    x_ = torch.Tensor()
    for elem in x:
        q = elem[0:7]
        c = torch.cat((elem[7:13], torch.zeros(1,dtype=torch.float32)))
        p = elem[13:].int().tolist()
        p_ = torch.zeros(7, dtype=torch.float32)
        for i in range(3):
            p_[i] = pos_dict[str(p[2*i]) + str(p[2*i+1])]
        for j in range(3, 6):
            p_[j] = j + 1
        elem_ = torch.cat((q, c, p_))
        x_ = torch.cat((x_, elem_.unsqueeze(0)))
    return x_


def transform_2d(x, repeat = [1, 1]):
    x = change_code(x)
    x = x.reshape(-1, 3, 7)
    x_flip = flip(x, dims = [2])
    x = torch.cat((x_flip, x), 2)
    x_1 = x
    for i in range(repeat[0] -1):
        x_1 = torch.cat((x_1, x), 1)
    x = x_1
    for i in range(repeat[1] -1):
        x = torch.cat((x, x_1), 2)
    return x.unsqueeze(1)


def transform_attention(x, repeat = [1, 1]):
    x = change_code(x)
    x = x.reshape(-1, 3, 7)
    x_1 = x
    for i in range(repeat[0] -1):
        x_1 = torch.cat((x_1, x), 1)
    x = x_1
    for i in range(repeat[1] -1):
        x = torch.cat((x, x_1), 2)
    pos = positional_encoding(repeat[1] * 7, 3)
    return x.transpose(1, 2) + pos


def positional_encoding(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * torch.div(i, 2, rounding_mode='floor')) / d_model)
    angle_rads = pos * angle_rates
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding
