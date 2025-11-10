import torch
from torch.utils.data import Dataset

class NFLPlayDataset(Dataset):
    def __init__(self, df_input, df_target, input_features, target_features,
                 player_side_embedding, player_position_embedding, player_role_embedding,
                 input_frames=5, output_frames=5):
        self.df_input = df_input
        self.df_target = df_target
        self.input_features = input_features
        self.target_features = target_features
        self.player_side_embedding = player_side_embedding
        self.player_position_embedding = player_position_embedding
        self.player_role_embedding = player_role_embedding
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.play_keys = list(df_input.groupby(['game_id','play_id','nfl_id']).groups.keys())
        
    def __len__(self):
        return len(self.play_keys)

    def __getitem__(self, idx):
        key = self.play_keys[idx]

        # Select the group
        df_in_group = self.df_input[
            (self.df_input['game_id'] == key[0]) &
            (self.df_input['play_id'] == key[1]) &
            (self.df_input['nfl_id'] == key[2])
        ].sort_values('frame_id')

        df_out_group = self.df_target[
            (self.df_target['game_id'] == key[0]) &
            (self.df_target['play_id'] == key[1]) &
            (self.df_target['nfl_id'] == key[2])
        ].sort_values('frame_id')

        # Skip plays that are too short
        if len(df_in_group) < self.input_frames or len(df_out_group) < self.output_frames:
            x_seq = torch.zeros(self.input_frames, len(self.input_features))
            y_seq = torch.zeros(self.output_frames, len(self.target_features))
            player_side_idx = torch.zeros(self.input_frames, dtype=torch.long)
            position_idx = torch.zeros(self.input_frames, dtype=torch.long)
            role_idx = torch.zeros(self.input_frames, dtype=torch.long)
            return x_seq, y_seq, player_side_idx, position_idx, role_idx

        # Continuous inputs
        x_seq = torch.tensor(df_in_group[self.input_features].iloc[-self.input_frames:].to_numpy(),
                             dtype=torch.float32)

        # Embeddings
        player_side_idx = torch.tensor(df_in_group['player_side_enc'].iloc[-self.input_frames:].to_numpy(),
                                       dtype=torch.long)
        position_idx = torch.tensor(df_in_group['position_idx'].iloc[-self.input_frames:].to_numpy(),
                                    dtype=torch.long)
        role_idx = torch.tensor(df_in_group['player_role_idx'].iloc[-self.input_frames:].to_numpy(),
                                dtype=torch.long)

        # Targets
        y_seq = torch.tensor(df_out_group[self.target_features].iloc[:self.output_frames].to_numpy(),
                             dtype=torch.float32)

        return x_seq, y_seq, player_side_idx, position_idx, role_idx
