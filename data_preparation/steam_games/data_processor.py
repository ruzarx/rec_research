

class SteamDataProcessor:
    def __init__(self):
        pass

    df = pd.read_csv('data/steam-200k.csv', names=['user', 'game', 'action', 'hours', 'sic'], header=None).drop(columns='sic').drop_duplicates()
n_games_played_per_user = (
    df[df['action'] == 'play'][['user', 'game', 'hours']]
    .groupby('user', as_index=False)
    .agg(
        games_played_by_user_count=('game', 'count'),
        hours_played_by_user_sum=('hours', 'sum'),
        hours_played_by_user_mean=('hours', 'mean'),
        hours_played_by_user_min=('hours', 'min'),
        hours_played_by_user_max=('hours', 'max'),
        hours_played_by_user_median=('hours', 'median'),
    )
)

n_games_bought_per_user = (
    df[df['action'] == 'purchase'][['user', 'game']]
    .groupby('user', as_index=False)
    .agg(
        games_bought_by_user_count=('game', 'count'),
    )
)

user_features = n_games_bought_per_user.merge(n_games_played_per_user, on='user', how='left').fillna(0)
user_features['games_played_to_bought_by_user_ratio'] = user_features['games_played_by_user_count'] / user_features['games_bought_by_user_count']
user_features


n_users_played_per_user = (
    df[df['action'] == 'play'][['user', 'game', 'hours']]
    .groupby('game', as_index=False)
    .agg(
        game_played_count=('user', 'count'),
        game_hours_sum=('hours', 'sum'),
        game_hours_mean=('hours', 'mean'),
        game_hours_min=('hours', 'min'),
        game_hours_max=('hours', 'max'),
        game_hours_median=('hours', 'median'),
    )
)

n_users_bought_per_user = (
    df[df['action'] == 'purchase'][['user', 'game']]
    .groupby('game', as_index=False)
    .agg(
        game_bought_count=('user', 'count'),
    )
)

game_features = n_users_bought_per_user.merge(n_users_played_per_user, on='game', how='left').fillna(0)
game_features['game_played_to_bought_ratio'] = game_features['game_played_count'] / game_features['game_bought_count']
game_features


df = pd.read_csv('data/steam-200k.csv', names=['user', 'game', 'action', 'hours', 'sic'], header=None).drop(columns='sic').drop_duplicates()

# games_count = df['game'].value_counts()
# good_games = games_count[games_count > 10].index
# df = df[df['game'].isin(good_games)]

# users_count = df['user'].value_counts()
# good_users = users_count[users_count > 10].index
# df = df[df['user'].isin(good_users)]

df = df.groupby(['user', 'game', 'action'], as_index=False)['hours'].sum()
df['n_event'] = df.groupby('user', as_index=False).cumcount() + 1

user_encoding = dict()
user_rev_encoding = dict()

n_user = 0
for user in df['user'].unique():
    user_encoding[int(user)] = n_user
    user_rev_encoding[n_user] = int(user)
    n_user += 1

game_encoding = dict()
game_rev_encoding = dict()

n_game = 0
for game in df['game'].unique():
    game_encoding[game] = n_game
    game_rev_encoding[n_game] = game
    n_game += 1

df['user'] = df['user'].replace(user_encoding)
df['game'] = df['game'].replace(game_encoding)
user_features['user'] = user_features['user'].replace(user_encoding)
game_features['game'] = game_features['game'].replace(game_encoding)
df['is_valid_play'] = False
df['is_valid_buy'] = False
df.loc[df[df['action'] == 'play'].sort_values(['user', 'n_event']).groupby('user').tail(1).index, 'is_valid_play'] = True
df.loc[df[df['action'] == 'purchase'].sort_values(['user', 'n_event']).groupby('user').tail(1).index, 'is_valid_buy'] = True

user_games = dict()
for _, user, game in df[['user', 'game']].itertuples():
    if user in user_games:
        user_games[user].add(game)
    else:
        user_games[user] = set([game])

val_pairs = dict()
all_games = df['game'].unique()
val_pair_users = []
val_pair_games = []
for user, games in user_games.items():
    for _ in range(10):
        game_candidate = random.choice(all_games)
        while game_candidate in games:
            game_candidate = random.choice(all_games)
        val_pair_users.append(user)
        val_pair_games.append(game_candidate)

negative_val_sample = pd.DataFrame({'user': val_pair_users, 'game': val_pair_games})
negative_val_sample['is_valid_buy'] = True
negative_val_sample['action'] = 'purchase'
negative_val_sample['hours'] = 0
print(negative_val_sample.shape)
print(df[df['action'] == 'purchase'].shape)
df = pd.concat([df, negative_val_sample])
print(df[df['action'] == 'purchase'].shape)

print(df['user'].nunique(), user_features['user'].nunique(), user_features.shape)

df = df.merge(user_features, on='user', how='left')
print(df.shape)
df = df.merge(game_features, on='game', how='left')
print(df.shape)

df = df[df['action'] == 'purchase']
df.shape, df['user'].nunique(), df['game'].nunique()




features_to_std = [col for col in user_features.columns.to_list() + game_features.columns.to_list() if col not in ['user', 'game', 'hours']]
feature_encoding = dict()
train_df = df[df['is_valid_buy'] == False]
valid_df = df[df['is_valid_buy'] == True]

for col in features_to_std:
    if col in train_df.columns:
        std = train_df[col].std()
        mean = train_df[col].mean()
        train_df[col] = (train_df[col] - mean) / std
        valid_df[col] = (valid_df[col] - mean) / std
        feature_encoding[col] = (std, mean)



user_cols = [col for col in user_features.columns if col not in ['user', 'game', 'hours']]
game_cols = [col for col in game_features.columns if col not in ['user', 'game', 'hours']]
train_dataset = SteamDataset(train_df, user_cols, game_cols)
valid_dataset = SteamDataset(valid_df, user_cols, game_cols)