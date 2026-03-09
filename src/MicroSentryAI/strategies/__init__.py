"""
Strategy Factory Module.

This module serves as the package entry point for model inference strategies.
It provides a factory function to instantiate the appropriate strategy class
(e.g., Anomalib) which abstracts the underlying model execution logic.
"""

from .anomalib_strategy import AnomalibStrategy


def load_strategy_from_folder(folder_path: str) -> AnomalibStrategy:
    """
    Factory function to obtain a model strategy instance.

    This function initializes the default strategy (AnomalibStrategy). The strategy
    instance is returned ready to load specific model weights via the UI controller.

    Args:
        folder_path (str): The directory containing the model artifacts.
                           (Reserved for future strategy selection logic based on folder contents).

    Returns:
        AnomalibStrategy: An instance of the inference strategy.
    """
    strategy = AnomalibStrategy()
    return strategy