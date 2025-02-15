
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import sucrose

sucrose.logger.setLevel("INFO")


class ExampleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


def main(index: int):
    sucrose.start_project('example', 'test1')

    model = ExampleModel(8, 16, 2)
    optim = Adam(model.parameters(), lr=1e-3)

    sucrose.load_state_dict(model=model, optim=optim)

    train_set = TensorDataset(torch.arange(100, dtype=torch.float32).reshape(10, 10))
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    eval_set = TensorDataset(torch.arange(50, dtype=torch.float32).reshape(5, 10))
    eval_loader = DataLoader(eval_set, batch_size=2)

    writer = sucrose.start_pytorch_tensorboard()

    for epoch in sucrose.epoch_range(10):
        print(f"Epoch {epoch}")
        model.train()

        for data, in train_loader:
            optim.step()
            sucrose.step()
            loss = torch.mean(data, dtype=torch.float32)
            writer.add_scalar('loss(train)', loss)

        model.eval()
        loss_list = []

        for data, in eval_loader:
            loss = torch.mean(data, dtype=torch.float32)
            loss_list.append(loss)

        num = len(loss_list)
        writer.add_scalar('loss(eval)', sum(loss_list)/num)

        sucrose.save_state_dict(epoch, model=model, optim=optim)


if __name__ == "__main__":
    main(0)
