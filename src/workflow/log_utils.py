import logging
import sys

loggers = {}

def get_logger(name):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        # logging configration
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger[name] = logger

        return logger

