import os
import json
import gc
import pandas as pd
from pandas.io.json import json_normalize

from log_utils import get_logger

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_PATH = os.path.join(RAW_PATH, 'processed_data')

RAW_TRAIN = os.path.join(RAW_PATH, 'train.csv')
RAW_TEST = os.path.join(RAW_PATH, 'test.csv')
FLAT_TRAIN = os.path.join(PROCESSED_PATH, 'train.csv')
FLAT_TEST = os.path.join(PROCESSED_PATH, 'test.csv')

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

logger = get_logger(__name__)


def process_data(nrows=None):
    """
    Process jsons in raw dataframes and save
    flatten dataframes for future use.
    :param nrows: process `nrows` rows in dataframes,
    if None, process all data
    """

    if os.path.isfile(FLAT_TRAIN) and os.path.isfile(FLAT_TEST):
        return None
    logger.debug('start processing')

    df = flatten_jsons_in_df(RAW_TRAIN, nrows)
    df.to_csv(FLAT_TRAIN, index=False)

    gc.enable()
    del df
    gc.collect()

    df = flatten_jsons_in_df(RAW_TEST, nrows)
    df.to_csv(FLAT_TEST, index=False)

    del df
    gc.collect()

    logger.debug('done processing')
    return None


def flatten_jsons_in_df(path_to_df, nrows=None):
    """
    Flatten individual dataframe
    Params: path_to_df, nrows
    return: flatenned dataframe
    """
    logger.debug(f'flatten json data at path: {path_to_df}')
    df = load_file(path_to_df,
                   nrows=nrows,
                   converters={column: json.loads for column in JSON_COLUMNS})

    for column in JSON_COLUMNS:
        tmp_df = json_normalize(df[column])
        tmp_df.columns = [
            f'{df[column].name}_{name}' for name in tmp_df.columns]
        df = df.drop(column, axis=1).merge(
            tmp_df, left_index=True, right_index=True)

    logger.debug('done flattening')
    return df


def load_data():
    """
    Load train, valid and test as dataframes
    Check if exists valid_file, if not make
    split
    Returns train, valid, test dataframes
    """

    test_path = os.path.join(PROCESSED_PATH, 'test.csv')
    train_path = os.path.join(PROCESSED_PATH, 'train.csv')
    valid_path = os.path.join(PROCESSED_PATH, 'valid.csv')

    if not os.path.isfile(valid_path):
        train_valid_split()

    trainset = load_file(train_path)
    validset = load_file(valid_path)
    testset = load_file(test_path)

    return trainset, validset, testset


def load_file(path, dtype={'fullVisitorId': 'str'}, nrows=None, **kwargs):
    """
    Load individual file into dataframe
    with "fullVisitorId" field as a 'str'
    Retutns: pandas dataframe
    """
    return pd.read_csv(path, dtype=dtype, nrows=nrows, **kwargs)


def train_valid_split():
    """
    TODO: implement
    """
    pass
