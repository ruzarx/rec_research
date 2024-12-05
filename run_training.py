










model = MatrixFactorization(1, df['user'].nunique(), df['game'].nunique(), len(user_cols), len(game_cols))

loss_function = torch.nn.BCELoss()
# pos_weight = torch.tensor([0.85])  # E.g., class_weight > 1 favors the positive class
# loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cpu')
train_model(model, optimizer, loss_function, train_dataloader, valid_dataloader, 8, device)