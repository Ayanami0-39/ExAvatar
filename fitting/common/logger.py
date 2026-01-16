


import logging
import os

INFO = '\033[94m'
DEBUG = '\033[96m'
OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING

class colorlogger():
    def __init__(self, log_dir = None, log_name='train_logs.txt'):
        # set log
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
                "{}%(asctime)s{} %(message)s".format(GREEN, END),
                "%m-%d %H:%M:%S")
        if log_dir is not None:
            log_file = os.path.join(log_dir, log_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_log = logging.FileHandler(log_file, mode='a')
            file_log.setLevel(logging.INFO)
            file_log.setFormatter(formatter)
            self._logger.addHandler(file_log)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.DEBUG)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(DEBUG + '[DEBUG] ' + END + str(msg))

    def info(self, msg):
        self._logger.info(INFO + '[INFO] ' + END + str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + '[WARNING] ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + '[CRITICAL] ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + '[ERROR] ' + str(msg) + END)