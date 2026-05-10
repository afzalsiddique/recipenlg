"""
Convert RecipeNLG CSV dataset to parquet shards for autoresearch.

Each recipe is formatted as plain text with Title, Ingredients, and Directions.
Shards are written to ~/.cache/autoresearch/data/ with a single "text" column,
matching the format expected by prepare.py's dataloader.

Usage:
    python convert_recipenlg.py
"""

import csv
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

from log_utils import setup_logger

logger = setup_logger("convert", "convert_recipenlg")

CSV_PATH = "/mmfs1/projects/PATH_TO_YOUR_DATASET/datasets/RecipeNLG/full_dataset.csv"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
RECIPES_PER_SHARD = 10000
SEED = 42


def format_recipe(title, ingredients_json, directions_json):
    try:
        ingredients = json.loads(ingredients_json)
    except (json.JSONDecodeError, TypeError):
        ingredients = []
    try:
        directions = json.loads(directions_json)
    except (json.JSONDecodeError, TypeError):
        directions = []

    parts = [f"Title: {title.strip()}"]
    if ingredients:
        parts.append(f"Ingredients: {' | '.join(ing.strip() for ing in ingredients)}")
    if directions:
        parts.append(f"Directions: {' | '.join(step.strip() for step in directions)}")
    return "\n".join(parts)


def convert_csv_to_shards():
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("shard_") and f.endswith(".parquet"))
    if existing:
        logger.info(f"Found {len(existing)} existing shards in {DATA_DIR}, skipping conversion")
        return len(existing) - 1

    logger.info(f"Reading CSV from {CSV_PATH}")
    recipes = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = format_recipe(row["title"], row["ingredients"], row["directions"])
            if text.strip():
                recipes.append(text)
            if (i + 1) % 500000 == 0:
                logger.info(f"  Read {i + 1} rows...")

    logger.info(f"Total recipes: {len(recipes)}")

    random.seed(SEED)
    random.shuffle(recipes)

    num_shards = (len(recipes) + RECIPES_PER_SHARD - 1) // RECIPES_PER_SHARD
    logger.info(f"Writing {num_shards} shards ({RECIPES_PER_SHARD} recipes per shard)")

    for shard_idx in range(num_shards):
        start = shard_idx * RECIPES_PER_SHARD
        end = min(start + RECIPES_PER_SHARD, len(recipes))
        shard_recipes = recipes[start:end]

        table = pa.table({"text": shard_recipes})
        shard_path = os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path)

        if (shard_idx + 1) % 50 == 0 or shard_idx == num_shards - 1:
            logger.info(f"  Written shard {shard_idx + 1}/{num_shards}")

    max_shard = num_shards - 1
    logger.info(f"Conversion complete. {num_shards} shards written to {DATA_DIR}")
    logger.info(f"MAX_SHARD = {max_shard} (last shard = validation)")
    return max_shard


if __name__ == "__main__":
    max_shard = convert_csv_to_shards()
    print(f"\nDone. MAX_SHARD = {max_shard}")
    print(f"Update prepare.py: MAX_SHARD = {max_shard}")
