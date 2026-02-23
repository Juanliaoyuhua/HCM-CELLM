import pandas as pd
import numpy as np
from config import cfg


def enrich_test_set(original_test_path, full_dataset_path, output_path):
    test_df = pd.read_parquet(original_test_path)
    full_df = pd.read_parquet(full_dataset_path)
    test_users = test_df[cfg.USER_ID_COL].unique()

    enriched_data = []

    for user_id in test_users:
        user_test_data = test_df[test_df[cfg.USER_ID_COL] == user_id]
        user_full_data = full_df[full_df[cfg.USER_ID_COL] == user_id]
        user_full_data = user_full_data.sort_values(by=cfg.TURN_COL)
        test_turns = user_test_data[cfg.TURN_COL].unique()

        if len(test_turns) > 0:
            min_turn = min(test_turns)
            user_enriched_data = user_full_data[user_full_data[cfg.TURN_COL] >= min_turn]
            enriched_data.append(user_enriched_data)
        else:
            enriched_data.append(user_full_data)

    if enriched_data:
        enriched_df = pd.concat(enriched_data, ignore_index=True)
    else:
        enriched_df = pd.DataFrame()

    enriched_df = enriched_df.sort_values(by=[cfg.USER_ID_COL, cfg.TURN_COL])
    enriched_df.to_parquet(output_path, index=False)
    analyze_enriched_testset(enriched_df)
    return enriched_df


def analyze_enriched_testset(df):
    user_groups = df.groupby(cfg.USER_ID_COL)
    user_stats = []

    for user_id, group in user_groups:
        turns = group[cfg.TURN_COL].tolist()
        min_turn = min(turns)
        max_turn = max(turns)
        turn_count = len(turns)

        user_stats.append({
            'user_id': user_id,
            'min_turn': min_turn,
            'max_turn': max_turn,
            'turn_count': turn_count
        })

    stats_df = pd.DataFrame(user_stats)


if __name__ == "__main__":
    from config import cfg

    enriched_df = enrich_test_set(
        original_test_path="data/test_queries.parquet",
        full_dataset_path="data/multiwoz22_converted.parquet",
        output_path="data/enriched_test_queries.parquet"
    )