from torchvision import transforms

import logging
import time
from configs import basic_config

import tqdm

frmt = basic_config.logger_config['format']
level = basic_config.logger_config['level']
log_filename = basic_config.logger_config['log_filename']
datefmt = basic_config.logger_config['datefmt']


def logger(log_filename, level, logger_filename):
    """
    This function is set the logger

    Parameters example:
        level = logging.DEBUG
        filename = '/content/drive/My Drive/GBN_l1_l2_Final_GridSearch_S_E_EVA5_S5_v6_FineTune_LR_scheduler_final_S6'
    """
    logging.basicConfig(format=frmt,
        datefmt=datefmt,
        level=level,
        filename=log_filename)
    logger = logging.getLogger(logger_filename)
    logger.addHandler(TqdmLoggingHandler())
    return logger

def get_logger(logger_filename):
    global log_filename
    log_filename = log_filename+"_"+time.ctime().replace(' ','_')+'_log.txt'
    return logger(log_filename, level, logger_filename)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record) 

if __name__ == "__main__":
    # from configs import basic_config
    # logger_config = basic_config.logger_config
    # print(logger_config)
    logger = get_logger(__name__)
    logger.debug("This is a debug log")
    logger.info("This is an info log")
    logger.critical("This is critical")
    logger.error("An error occurred")
