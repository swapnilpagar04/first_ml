import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
LOG_FILE = f"{LOG_DIR}/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",  # append mode
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Create a logger object
logger = logging.getLogger(__name__)
