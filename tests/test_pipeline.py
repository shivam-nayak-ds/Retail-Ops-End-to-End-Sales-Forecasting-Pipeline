import pytest
import pandas as pd
from Retail_Ops_Pipeline.components.data_validation import DataValidation
from Retail_Ops_Pipeline.config.configuration import ConfigurationManager

def test_data_validation_initialization():
    """
    Tests if the DataValidation component can be initialized correctly.
    """
    try:
        config_manager = ConfigurationManager()
        validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=validation_config)
        assert data_validation.config is not None
    except Exception as e:
        pytest.fail(f"DataValidation initialization failed: {e}")

def test_config_retrieval():
    """
    Tests if ConfigurationManager retrieves the correct config types.
    """
    config_manager = ConfigurationManager()
    ingestion_config = config_manager.get_data_ingestion_config()
    assert ingestion_config.root_dir is not None
