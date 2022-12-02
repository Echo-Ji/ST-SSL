import os
import logging
from datetime import datetime
import pandas as pd 

def get_logger(root, name=None, debug=True):
    # when debug is true, show DEBUG and INFO in screen
    # when debug is false, show DEBUG in file and info in both screen&file
    # INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger

class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)

if __name__ == '__main__':
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log.txt', debug=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')