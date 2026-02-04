import logging
import sys


def create_logger(logdir: str = "") -> logging.Logger:
    head = "%(asctime)-15s %(message)s"
    if logdir:
        logging.basicConfig(filename=logdir, format=head)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


class AverageMeter:
    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


__all__ = ["create_logger", "AverageMeter"]

