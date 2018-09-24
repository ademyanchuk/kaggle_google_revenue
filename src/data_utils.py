import os
import json
import gc
import pandas as pd
from pandas.io.json import json_normalize

from cleaner import Cleaner
from log_utils import get_logger

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_PATH = os.path.join(RAW_PATH, 'processed_data')

RAW_TRAIN = os.path.join(RAW_PATH, 'train.csv')
RAW_TEST = os.path.join(RAW_PATH, 'test.csv')
FLAT_TRAIN = os.path.join(PROCESSED_PATH, 'train.csv')
FLAT_TEST = os.path.join(PROCESSED_PATH, 'test.csv')
PROC_TRAIN = os.path.join(PROCESSED_PATH, 'processed_train.csv')
PROC_TEST = os.path.join(PROCESSED_PATH, 'processed_test.csv')

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
TARGET_COL_NAME = 'totals_transactionRevenue'

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
    df = load_csv(path_to_df,
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
    Load train and test as dataframes
    Returns train, test dataframes
    """

    test_path = os.path.join(PROCESSED_PATH, 'test.csv')
    train_path = os.path.join(PROCESSED_PATH, 'train.csv')

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    return train_df, test_df


def load_csv(path, dtype={'fullVisitorId': 'str'}, nrows=None, **kwargs):
    """
    Load individual file into dataframe
    with "fullVisitorId" field as a 'str'
    Retutns: pandas dataframe
    """
    return pd.read_csv(path, dtype=dtype, nrows=nrows, **kwargs)


def align_frames(train_df, test_df, target_name=TARGET_COL_NAME):
    """
    Align train_df on test_df
    If column in train_df and
    not in test_df, it'll be
    removed from train_df
    """
    target_col = train_df[target_name]
    train_df = train_df.drop(target_name, axis=1)
    test_df, train_df = test_df.align(train_df, join='inner', axis=1)
    train_df[target_name] = target_col
    return train_df, test_df

def save_processed_to_csv(train_df, test_df):
    """
    Save fully processed train_df and
    test_df to scv's in special
    processed folder
    """
    train_df.to_csv(PROC_TRAIN, index=False)
    test_df.to_csv(PROC_TEST, index=False)

def preprocess_pipeline(nrows=None, nan_fraction=1):
    """
    Full preprocessing pipeline
    with intermidiate and final
    data saving

    :param nan_fraction: if fraction
    of Nan in data_frame columns >= `nan_fraction`
    removes those columns (default = 1, remove
    only fully Nan columns)
    :param nrows: process only `nrows` of data
    (default None, process all data)

    Returns processed and cleaned train and test
    data frames
    """
    logger.debug('Preprocessing pipeline started..')
    # flatten jsons inside data_frame columns
    process_data(nrows=nrows)

    logger.debug('loading flatenned data..')
    train_df, test_df = load_data()

    cleaner = Cleaner()

    # cleaning step
    logger.debug('cleaning data..')
    train_df = cleaner.clean(train_df, nan_fraction)
    test_df = cleaner.clean(test_df, nan_fraction)

    logger.debug('aligning data..')
    train_df, test_df = align_frames(train_df, test_df)

    logger.debug('finally saving preprocessed data')
    save_processed_to_csv(train_df, test_df)

    logger.debug('Preprocessing pipeline - done.')
    return train_df, test_df


def train_valid_split():
    """
    TODO: implement
    """
    pass
