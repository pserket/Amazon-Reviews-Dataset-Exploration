# Pete Sheurpukdi and Chris Wynne
# Created with jupyter notebook and run on AWS cluster.

import dask.dataframe as dd
from dask.distributed import Client

import pandas as pd
import numpy as np
import ast
import json
from datetime import datetime
from itertools import chain

# series of transformations to produce the users table as dataframe
def transform_user(user_reviews_csv):
    client = Client('127.0.0.1:8786')
    # client = client.restart()

    def find_helpful(df):
        return df['helpful'].apply(clean_helpful)

    def clean_helpful(x):
        try:
            t = ast.literal_eval(x)
            return int(t[0])
        except:
            return 0

    def find_total(df):
        return df['helpful'].apply(clean_total)

    def clean_total(x):
        try:
            t = ast.literal_eval(x)
            return int(t[1])
        except:
            return 0

    def get_year(df):
        return df['reviewTime'].apply(clean_year)

    def clean_year(x):
        try:
            return int(x[-4:])
        except:
            return 9999

    def task1(test_df, parts):
        n_part = parts
        total_votes = test_df.map_partitions(find_total)
        only_helpful = test_df.map_partitions(find_helpful)
        reviewing_since = test_df.map_partitions(get_year)
        test_df['reviewing_since'] = reviewing_since
        test_df['total_votes'] = total_votes
        test_df['helpful_votes'] = only_helpful
        rev_ids = test_df.groupby('reviewerID').agg({'asin': 'count',
                                                     'overall': 'mean',
                                                     'reviewing_since': 'min',
                                                     'helpful_votes': 'sum',
                                                     'total_votes': 'sum'
                                                     }, split_out=n_part).reset_index(
        ).rename(columns={'asin': 'number_products_rated', 'overall': 'avg_rating'})
        return rev_ids

    parts = 8
    results = task1(user_reviews_csv, parts)
    submit = results.describe().compute().round(2)
    with open('results_1A.json', 'w') as outfile:
        json.dump(json.loads(submit.to_json()), outfile)


def compute_stats(user_reviews_csv, products_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()

    rev_ids = user_reviews_csv['asin'].persist()
    id_list = products_csv['asin'].persist()

    # Q1
    prod_prop = (products_csv.isnull().mean() * 100).compute().round(2)
    rev_prop = (user_reviews_csv.isnull().mean() * 100).compute().round(2)

    # Q2
    corr_df = user_reviews_csv[['asin', 'overall']].merge(products_csv[['asin', 'price']], on='asin')
    corrSeries = corr_df[['overall', 'price']].corr().compute()

    # Q3
    priceStats = products_csv.price.describe(percentiles=[0.5]).compute()

    # Q4
    def category_eval(df):
        return df['categories'].apply(clean_category)

    def clean_category(val):
        try:
            return ast.literal_eval(val)[0][0]
        except:
            return val

    clean_cats = products_csv.map_partitions(category_eval)
    category_counts = clean_cats.value_counts().compute()

    # Q5

    def q5check_ids_exist(df, id_list):

        for i in df.to_frame().iterrows():
            id_to_check = list(i[1])[0]
            if not (id_to_check in id_list):
                return 1
        return 0

    q5_ans = q5check_ids_exist(rev_ids, id_list)

    # Q6

    def related_eval(df):
        return df['related'].apply(related_to_prod_list)

    def related_to_prod_list(related_dict):
        try:
            related_dict = ast.literal_eval(related_dict)
            return list(chain(*related_dict.values()))
        except:
            return related_dict

    def q6check_ids_exist(df, id_list):
        for i in df.iterrows():
            idList = list(i[1])[0]
            if type(idList) == list:
                for related_id in idList:
                    if not (related_id in id_list):
                        return 1
        return 0

    flatRelatedIds = products_csv.map_partitions(related_eval).to_frame()
    q6_ans = q6check_ids_exist(flatRelatedIds, id_list)

    out_dict = {"q1": {"products": dict(prod_prop), "reviews": dict(rev_prop)},
                "q2": corrSeries['overall']['price'].round(2),
                "q3": dict(priceStats.drop('count').round(2)),
                "q4": dict(category_counts),
                "q5": q5_ans,
                "q6": q6_ans}

    def convert(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    with open('results_1B.json', 'w') as outfile:
        json.dump(out_dict, outfile, default=convert)

    # Write results to "results_1B.json" here and round solutions to 2 decimal points
