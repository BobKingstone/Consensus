"""Config Util Class"""

import configparser


CONFIG_FILE = 'client_config.ini'
""" File containing client settings."""

class ConfigHelper():
    """Config helper class"""
    @staticmethod
    def get_config_section_values(section, val):
        """Returns the requested values"""
        config = configparser.ConfigParser()
        config.read( CONFIG_FILE)
        return config.get(section, val)
