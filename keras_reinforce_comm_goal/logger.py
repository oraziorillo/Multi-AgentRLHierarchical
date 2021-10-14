import os
import logging

def setup_logger(name, log_file, formatter, mode='a', level=logging.INFO):
    """To setup a new logger"""

    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Parameters
log_dir = './temp/' # log directory

# Create logs directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s -- %(message)s',
                    datefmt='%m-%d %H:%M')

# Set a format for our logs
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s -- %(message)s', datefmt='%m-%d %H:%M')

# info file logger
info_logger = setup_logger('info', log_dir+'info.log', formatter)

# results file logger
results_logger = setup_logger('results', log_dir+'results.log', formatter)

# env file logger
env_logger = setup_logger('env', log_dir+'env.log', formatter)

# Define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
