import os
import sys
import subprocess

from utils import logger_utils
logger = logger_utils.get_logger(__name__)


orchestrator = sys.argv[1]

logger.info(f"Started Executing the orchestrator {orchestrator}.py")
print(f"Started Executing the orchestrator {orchestrator}.py")
# os.system(f"python ./orchestrators/{orchestrator}.py")
orhcestrator_execution = subprocess.Popen(["python", f"./orchestrators/{orchestrator}.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, errors = orhcestrator_execution.communicate()
orhcestrator_execution.wait()
print(output)
print(errors)
logger.info(f"Ended Executing the orchestrator {orchestrator}.py")
print(f"Ended Executing the orchestrator {orchestrator}.py")

# print(os.curdir)
# print('getcwd:      ', os.getcwd())
# print('__file__:    ', __file__)

