__author__ = 'Chi'

import logging
import time


#constract the logging function
def loginformation(filename, logginglevel, loggingmessage):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=filename,
                        filemode='w')
    if(logginglevel == 'info'):
        logging.info(loggingmessage)
    elif(logginglevel == 'debug'):
        logging.debug(loggingmessage)
    else:
        logging.warning(loggingmessage)
