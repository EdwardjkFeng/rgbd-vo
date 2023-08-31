import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler and set level to info 
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
