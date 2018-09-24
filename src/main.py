from data_utils import preprocess_pipeline
from log_utils import get_logger

logger = get_logger(__name__)

def make_pipeline():
    """
    TODO: implement
    """
    logger.debug("Pipeline started..")
    train_df, test_df = preprocess_pipeline(nrows=10000)
    # make_feature_matrix(train_set, valid_set, test_set)
    # train_model()
    # validate_model()
    # make_submission()
    # save_model_info_to_csv()
    logger.debug('Done.')


def make_feature_matrix(train_set, valid_set, test_set):
    """
    TODO: implement
    """
    pass


def train_model():
    """
    TODO: implement
    """
    pass


def validate_model():
    """
    TODO: implement
    """
    pass


def make_submission():
    """
    TODO: implement
    """
    pass


def save_model_info_to_csv():
    """
    TODO: implement
    """
    pass


if __name__ == '__main__':
    make_pipeline()
