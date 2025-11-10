import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_nfl_data(data_dir='../Data', week_end=2):
    """
    Load NFL input/output CSVs, split into train/val, compute deltas,
    normalize, and prepare categorical encoding for player_side.
    Fully vectorized for speed.
    """
    WEEKS_ALL = [f'w{i:02d}' for i in range(1, 19)]

    input_data, target_data = [], []

    print("Loading CSVs...")
    for i, week_code in enumerate(WEEKS_ALL):
        week_num = i + 1
        file_in = os.path.join(data_dir, f'input_2023_{week_code}.csv')
        file_out = os.path.join(data_dir, f'output_2023_{week_code}.csv')

        if os.path.exists(file_in):
            df_in = pd.read_csv(file_in)
            df_in['week'] = week_num
            input_data.append(df_in)
            print(f"Loaded {file_in}, shape: {df_in.shape}")
        if os.path.exists(file_out):
            df_out = pd.read_csv(file_out)
            df_out['week'] = week_num
            target_data.append(df_out)
            print(f"Loaded {file_out}, shape: {df_out.shape}")

    if not input_data or not target_data:
        raise FileNotFoundError(f"No input/output CSVs found in {data_dir}")

    df_input_all = pd.concat(input_data, ignore_index=True)
    df_target_all = pd.concat(target_data, ignore_index=True)
    print(f"Total input shape: {df_input_all.shape}, total target shape: {df_target_all.shape}")

    df_input_all = df_input_all[df_input_all['player_to_predict'] == True]

    # --- Normalize play direction ---
    left_mask = df_input_all['play_direction'].str.lower() == 'left'
    df_input_all.loc[left_mask, 'x'] = 120 - df_input_all.loc[left_mask, 'x']
    df_input_all.loc[left_mask, 'o'] = (360 - df_input_all.loc[left_mask, 'o']) % 360
    df_input_all.loc[left_mask, 'dir'] = (360 - df_input_all.loc[left_mask, 'dir']) % 360
    df_input_all.loc[left_mask, 'absolute_yardline_number'] = 120 - df_input_all.loc[left_mask, 'absolute_yardline_number']

    # Identify left plays in the input dataframe
    left_plays = df_input_all.loc[df_input_all['play_direction'].str.lower() == 'left', ['game_id', 'play_id']].drop_duplicates()

    # Create a mask for the output dataframe
    left_mask_output = df_target_all.merge(left_plays, on=['game_id', 'play_id'], how='left', indicator=True)['_merge'] == 'both'

    # Apply the same transformations
    df_target_all.loc[left_mask_output, 'x'] = 120 - df_target_all.loc[left_mask_output, 'x']

    print(f"Play direction normalization done. Left plays: {left_mask.sum()}")

    # --- Encode player_side ---
    df_input_all['player_side_enc'] = df_input_all['player_side'].str.lower().map({'offense': 0, 'defense': 1})
    df_input_all['player_side_enc'] = df_input_all['player_side_enc'].fillna(0).astype(int)

    # --- Compute last input positions ---
    last_positions = df_input_all.sort_values('frame_id').groupby(
        ['game_id','play_id','nfl_id']
    ).last()[['x','y']].rename(columns={'x':'x_last_input','y':'y_last_input'})

    # Merge into targets
    df_target_all = df_target_all.merge(last_positions, on=['game_id','play_id','nfl_id'], how='left')

    # ---- Encode position --
    positions = df_input_all['player_position'].unique()
    pos2idx = {pos: i for i, pos in enumerate(positions)}
    df_input_all['position_idx'] = df_input_all['player_position'].map(pos2idx)

    # ---- Encode role --
    roles = df_input_all['player_role'].unique()
    role2idx = {r: i for i, r in enumerate(roles)}
    df_input_all['player_role_idx'] = df_input_all['player_role'].map(role2idx)

    # --- Compute deltas (vectorized) ---
    df_target_all = df_target_all.sort_values(['game_id','play_id','nfl_id','frame_id'])
    df_target_all[['delta_x','delta_y']] = df_target_all.groupby(
        ['game_id','play_id','nfl_id']
    )[['x','y']].diff().fillna(0)

    # Correct first frame delta using last input position
    first_frame_mask = df_target_all.groupby(
        ['game_id','play_id','nfl_id']
    ).cumcount() == 0
    df_target_all.loc[first_frame_mask, 'delta_x'] = df_target_all.loc[first_frame_mask, 'x'] - df_target_all.loc[first_frame_mask, 'x_last_input']
    df_target_all.loc[first_frame_mask, 'delta_y'] = df_target_all.loc[first_frame_mask, 'y'] - df_target_all.loc[first_frame_mask, 'y_last_input']

    # --- Split train / val ---
    df_input_train = df_input_all[df_input_all['week'] <= week_end].copy()
    df_target_train = df_target_all[df_target_all['week'] <= week_end].copy()
    df_input_val = df_input_all[df_input_all['week'] > 17].copy()
    df_target_val = df_target_all[df_target_all['week'] > 17].copy()
    print(f"Train/val split: train={df_input_train.shape}, val={df_input_val.shape}")

    # --- Scale continuous features ---
    linear_scaler = StandardScaler()
    df_input_train[['s','a']] = linear_scaler.fit_transform(df_input_train[['s','a']])
    df_input_val[['s','a']] = linear_scaler.transform(df_input_val[['s','a']])

    delta_scaler = StandardScaler()     # for delta_x, delta_y
    yards_scaler = StandardScaler()
    pos_scaler = StandardScaler()      # for absolute yards

    df_input_train[['x','y']] = pos_scaler.fit_transform(df_input_train[['x','y']])
    df_input_val[['x','y']] = pos_scaler.transform(df_input_val[['x','y']])
    df_input_train['absolute_yardline_number'] = yards_scaler.fit_transform(df_input_train[['absolute_yardline_number']])
    df_input_val['absolute_yardline_number'] = yards_scaler.transform(df_input_val[['absolute_yardline_number']])

    df_target_train[['delta_x','delta_y']] = delta_scaler.fit_transform(df_target_train[['delta_x','delta_y']])
    df_target_val[['delta_x','delta_y']] = delta_scaler.transform(df_target_val[['delta_x','delta_y']])

    # --- Encode directions as sin/cos ---
    for col in ['dir','o']:
        df_input_train[f'{col}_sin'] = np.sin(np.deg2rad(df_input_train[col]))
        df_input_train[f'{col}_cos'] = np.cos(np.deg2rad(df_input_train[col]))
        df_input_val[f'{col}_sin'] = np.sin(np.deg2rad(df_input_val[col]))
        df_input_val[f'{col}_cos'] = np.cos(np.deg2rad(df_input_val[col]))

    return df_input_train, df_target_train, df_input_val, df_target_val, pos_scaler, delta_scaler, positions, roles
