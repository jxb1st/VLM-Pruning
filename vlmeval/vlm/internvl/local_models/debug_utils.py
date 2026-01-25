"""
Debug logger utilities for local InternVL models.

This module provides a global debug logger interface that can be used
by local model code without creating dependencies on VLMEvalKit.
"""

_global_debug_logger = None


def set_debug_logger(logger):
    """Set the global debug logger for local models.
    
    Args:
        logger: Debug logger instance from VLMEvalKit
    """
    global _global_debug_logger
    _global_debug_logger = logger


def get_debug_logger():
    """Get the global debug logger instance.
    
    Returns:
        Debug logger instance or None if not set
    """
    return _global_debug_logger

