import torch.nn


class MatrixFactorization(torch.nn.Module):
    def __init__(self, model_size: int, n_users: int, n_items: int):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(n_users + 1, model_size, padding_idx=0)
        self.item_embedding = torch.nn.Embedding(n_items + 1, model_size, padding_idx=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.user_bias = torch.nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.item_bias = torch.nn.Embedding(n_items + 1, 1, padding_idx=0)
        return

    def forward(self, inputs: tuple):
        user_ids, item_ids, _, _ = inputs
        x = torch.sum(self.user_embedding(user_ids) * self.item_embedding(item_ids), dim=-1)
        x = x + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze()
        return self.sigmoid(x)
    