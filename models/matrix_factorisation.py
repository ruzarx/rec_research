import torch
import torch.nn.functional as F


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
    

class MatrixFactorizationBPR(torch.nn.Module):
    def __init__(self, model_size, num_users, num_items):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users + 1, model_size, padding_idx=0)
        self.item_embedding = torch.nn.Embedding(num_items + 1, model_size, padding_idx=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.user_bias = torch.nn.Embedding(num_users + 1, 1, padding_idx=0)
        self.item_bias = torch.nn.Embedding(num_items + 1, 1, padding_idx=0)

    def forward(self, user, pos_item, neg_item):
        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        pos_item_emb = F.normalize(pos_item_emb, p=2, dim=1)
        neg_item_emb = F.normalize(neg_item_emb, p=2, dim=1)

        user_bias = self.user_bias(user).squeeze()
        pos_item_bias = self.item_bias(pos_item).squeeze()
        neg_item_bias = self.item_bias(neg_item).squeeze()
        return user_emb, pos_item_emb, neg_item_emb, user_bias, pos_item_bias, neg_item_bias


# def bpr_loss(user_emb, pos_item_emb, neg_item_emb, user_bias, pos_item_bias, neg_item_bias):
#     pos_scores = (user_emb * pos_item_emb).sum(dim=1) + user_bias + pos_item_bias
#     neg_scores = (user_emb * neg_item_emb).sum(dim=1) + user_bias + neg_item_bias
#     loss = -F.logsigmoid(pos_scores - neg_scores).mean()
#     return loss

def bpr_loss(user_emb, pos_item_emb, neg_item_emb, user_bias, pos_item_bias, neg_item_bias):
    # Compute scores for positive and negative items
    pos_scores = torch.einsum('ij,ij->i', user_emb, pos_item_emb) + user_bias.squeeze() + pos_item_bias.squeeze()
    neg_scores = torch.einsum('ij,ij->i', user_emb, neg_item_emb) + user_bias.squeeze() + neg_item_bias.squeeze()
    
    # Compute BPR loss
    margin = pos_scores - neg_scores
    loss = -F.logsigmoid(margin).mean()
    return loss
