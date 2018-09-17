from data_utils import process_data, load_data
from log_utils import get_logger

logger = get_logger(__name__)


def make_pipeline():
    """
    TODO: implement
    """
    logger.debug("Started..")
    process_data(nrows=10000)
    # train_set, valid_set, test_set = load_data()
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
