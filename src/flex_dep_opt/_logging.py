"""Console logging setup for the command line. Library code only emits log records."""

import logging

from tqdm.auto import tqdm


class _ConsoleFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return message if record.levelno == logging.INFO else f"{record.levelname}: {message}"


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if not any(isinstance(h, _TqdmLoggingHandler) for h in root_logger.handlers):
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        handler = _TqdmLoggingHandler()
        handler.setFormatter(_ConsoleFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    logging.getLogger("pyomo").setLevel(logging.WARNING)
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
