
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import sucrose


class ExampleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


def main(index: int):
    sucrose.start_project('example', 'test1')
    config_data = sucrose.read_config()

    model = ExampleModel()
    optim = Adam(model.parameters(), lr=1e-3)

    data_loaded = sucrose.load_ckpts()

    train_set = TensorDataset(torch.arange(100, dtype=torch.float32).reshape(10, 10))
    train_loader = DataLoader(train_set, shuffle=True)
    eval_set = TensorDataset(torch.arange(50, dtype=torch.float32).reshape(5, 10))
    eval_loader = DataLoader(eval_set)

    writer = sucrose.start_pytorch_tensorboard()

    for epoch in range(100):
        model.train()

        for data in train_loader:

            ...
            optim.step()
            sucrose.step()
            writer.add_scalar()

        model.eval()

        for data in eval_loader:
            ...

        writer.add_scalar()

        sucrose.save_ckpts(epoch, {'model': model.state_dict()})
