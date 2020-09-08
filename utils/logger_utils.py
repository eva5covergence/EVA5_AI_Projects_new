import logging
import time
from configs import basic_config

def logger(log_filename, level, logger_filename):
    """
    This function is set the logger

    Parameters example:
        level = logging.DEBUG
        filename = '/content/drive/My Drive/GBN_l1_l2_Final_GridSearch_S_E_EVA5_S5_v6_FineTune_LR_scheduler_final_S6'
    """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=level,
        filename=log_filename)
    logger = logging.getLogger(logger_filename)
    return logger

def get_logger(logger_filename):
    logger_config = basic_config.logger_config
    return logger(logger_config['log_filename'], logger_config['level'], logger_filename)



if __name__ == "__main__":
    from configs import basic_config
    logger_config = basic_config.logger_config
    print(logger_config)
    logger = logger(logger_config['log_filename'], logger_config['level'], __name__)
    logger.debug("This is a debug log")
    logger.info("This is an info log")
    logger.critical("This is critical")
    logger.error("An error occurred")