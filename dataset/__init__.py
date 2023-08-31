from utils.tools import get_class
from .base_dataset import BaseDataset
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler and set level to info 
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

def get_dataset(name):
    return get_class(name, __name__, BaseDataset)