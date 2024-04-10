from federatedscope.register import register_model
from torch import nn
import argparse


class FedGRU(nn.Module):
    def __init__(self, input_size, output_size, args):
        super(FedGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, num_layers=args.num_layers, hidden_size=args.hidden_size,
                          batch_first=True)
        self.linear = nn.Linear(args.hidden_size, output_size)

    def forward(self, x):  # B x T x N x D
        output, _ = self.gru(x[..., 0])
        return self.linear(output[:, -1:, :]).unsqueeze(3)


def load_my_net(data_shape):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--hidden_size', default=50)
    args = parser.parse_args(args=list())
    return FedGRU(data_shape[2], data_shape[2], args)


def call_my_net(model_config, data_shape):  # data (B x T x N x D)
    if model_config.type == "fedgru":
        return load_my_net(data_shape)


register_model("fedgru", call_my_net)
