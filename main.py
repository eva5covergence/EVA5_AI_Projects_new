import os
import sys

from utils import logger_utils
logger = logger_utils.get_logger(__name__)


orchestrator = sys.argv[1]

logger.info(f"Started Executing the orchestrator {orchestrator}.py")
print(f"Started Executing the orchestrator {orchestrator}.py")
os.system(f"python ./orchestrators/{orchestrator}.py")
logger.info(f"Ended Executing the orchestrator {orchestrator}.py")
print(f"Ended Executing the orchestrator {orchestrator}.py")

# print(os.curdir)
# print('getcwd:      ', os.getcwd())
# print('__file__:    ', __file__)

