import logging

def configure_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

configure_logging('app.log')

logger = logging.getLogger(__name__)