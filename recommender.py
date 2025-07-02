# recommender.py
import psycopg2
import pandas as pd
from psycopg2 import sql
import json
import re
from config import DB_CONFIG

class ProductRecommender:
    def __init__(self):
        self.conn = psycopg2.connect(
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            sslmode="require"
        )
        self.cursor = self.conn.cursor()
        self.schema = DB_CONFIG["schema"]
        self.df = None
        self._set_schema()

    def _set_schema(self):
        self.cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        self.cursor.execute("SELECT current_schema()")
        print("Schema set to:", self.cursor.fetchone()[0])

    def load_data(self, partner_id, category=None, min_qty=1):
        table = "Products"
        query = sql.SQL("SELECT * FROM {} WHERE partner_id = %s").format(sql.Identifier(table))
        self.cursor.execute(query, [partner_id])
        df = pd.DataFrame(self.cursor.fetchall(), columns=[desc[0] for desc in self.cursor.description])

        if category:
            df = df[df['category'] == category]
        df = df[df['quantity'] >= min_qty]

        self.df = df.reset_index(drop=True)
        return self.df

    def extract_color_from_name(self):
        def split_colors(name):
            try:
                parts = name.split(',')
                if len(parts) >= 3:
                    return [c.strip() for c in parts[1].strip().split('/')]
                return []
            except:
                return []

        self.df['color_from_name'] = self.df['product_name'].fillna('').apply(split_colors)

    def expand_options(self):
        def parse_options(option_str):
            try:
                option_dict = json.loads(option_str) if isinstance(option_str, str) else option_str
                return {k: v[0] if isinstance(v, list) else v for k, v in option_dict.items()}
            except:
                return {}

        options_df = self.df['options'].apply(parse_options).apply(pd.Series)
        self.df = pd.concat([self.df.drop(columns=['Size', 'Color', 'Width', 'Model'], errors='ignore'), options_df], axis=1)

    def extract_first_word(self):
        self.df['first_word'] = self.df['product_name'].fillna('').str.split().str[0]

    def extract_department(self):
        def get_department(name):
            match = re.search(r"\b(Women's|Men's|Unisex|Kids')\b", str(name))
            return match.group(1) if match else 'Unknown'
        self.df['Department'] = self.df['product_name'].apply(get_department)

    def expand_metadata(self):
        keys = ["custom.color", "custom.model", "google.gender", "my_fields.size", "my_fields.width"]
        def extract_keys(meta_str):
            try:
                meta_dict = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                return {k: meta_dict.get(k) for k in keys}
            except:
                return {k: None for k in keys}

        meta_df = self.df['metadata'].apply(extract_keys).apply(pd.Series)
        self.df = pd.concat([self.df, meta_df], axis=1)

    def preprocess(self):
        self.extract_color_from_name()
        self.expand_options()
        self.extract_first_word()
        self.extract_department()
        self.expand_metadata()
        self.df = self.df.rename(columns={
            'first_word': 'first_word_from_name',
            'Department': 'gender_from_name'
        })

    def recommend(self, gender, size, width=None, brands=None, colors=None, top_k=10):
        df = self.df.copy()
        width = width or ''
        import numpy as np

        width_map = {
            'narrow': {'exact': ['narrow'], 'compatible': ['medium (regular)', 'regular']},
            'medium': {'exact': ['medium (regular)', 'regular'], 'compatible': []},
            'wide': {'exact': ['wide'], 'compatible': ['medium (regular)', 'extra wide']},
            'extra wide': {'exact': ['extra wide'], 'compatible': ['wide']}
        }

        def parse_size(size_str):
            try:
                if '-' in size_str:
                    low, high = map(float, size_str.replace('.', '').split('-'))
                    return low - 0.5, high + 0.5, True
                val = float(size_str.replace('.', ''))
                return val - 0.5, val + 0.5, False
            except:
                return None, None, False

        df[['size_min', 'size_max', 'is_range']] = df['my_fields.size'].apply(lambda x: pd.Series(parse_size(x)))
        df = df[df['gender_from_name'].str.lower() == gender.lower()]
        df = df[(df['size_min'] <= float(size)) & (df['size_max'] >= float(size))]

        def compute_score(row):
            score = 0
            if row['is_range']:
                score += 18.75
            else:
                diff = abs(float(row['size_min']) + 0.5 - float(size))
                score += 31.25 if diff < 0.01 else 21.875 if diff == 0.5 else 0

            w = str(row.get('my_fields.width', '')).lower()
            if width.lower() in width_map:
                if w in width_map[width.lower()]['exact']:
                    score += 12.5
                elif w in width_map[width.lower()]['compatible']:
                    score += 6.25

            vendor = str(row['vendor']).lower()
            model = str(row.get('custom.model', '')).lower()
            for brand, prefs in (brands or {}).items():
                if brand.lower() == vendor:
                    score += 25
                    if 'models' in prefs and any(m.lower() in model for m in prefs['models']):
                        score += 25

            color = str(row.get('custom.color', '')).lower().split('/')
            for i, c in enumerate(color):
                if c in [clr.lower() for clr in (colors or [])]:
                    score += 6.25 - i * 1.25
                    break

            return score

        df['score'] = df.apply(compute_score, axis=1)
        return df.sort_values(by=['score', 'quantity'], ascending=[False, False]).head(top_k)
