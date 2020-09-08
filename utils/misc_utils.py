import torch
import sys
from utils.logger_utils import get_logger

def set_manual_seed(seed):
    logger = get_logger(__name__)
    cuda = is_cuda()
    logger.info(f"CUDA Available? {cuda}")
    torch.cuda.manual_seed(seed) if cuda else torch.manual_seed(seed)


def is_cuda():
    return torch.cuda.is_available()

def get_device_type():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

if __name__ == "__main__":
    set_manual_seed(20)
    print(is_cuda())