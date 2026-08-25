import logging
import sys
from flask import request, g, has_request_context

class RequestIdFilter(logging.Filter):
    """Injects the request ID into the log record if available."""
    def filter(self, record):
        if has_request_context():
            record.request_id = getattr(g, 'request_id', 'N/A')
        else:
            record.request_id = 'SYSTEM'
        return True

def setup_logger(name: str = "dr_system") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [ReqID: %(request_id)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addFilter(RequestIdFilter())
        
    return logger

logger = setup_logger()
