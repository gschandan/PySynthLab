import logging
import unittest
from unittest.mock import patch, mock_open
import argparse
import yaml
from src.cegis.z3.options import Options
from src.cegis.z3.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.default_options = Options()

    def test_generate_argparse_from_options(self):
        parser = ConfigManager.generate_argparse_from_options()
        self.assertIsInstance(parser, argparse.ArgumentParser)

        args = parser.parse_args(['--synthesisparameters__max_depth', '6'])
        self.assertEqual(args.synthesisparameters__max_depth, 6)

        args = parser.parse_args(['--logging__level', 'INFO'])
        self.assertEqual(args.logging__level, 'INFO')

    def test_setup_logger(self):
        ConfigManager.setup_logger()
        self.assertEqual(ConfigManager.logger.level, logging.NOTSET)
        self.assertTrue(ConfigManager.logger.hasHandlers())

    @patch('builtins.open', new_callable=mock_open, read_data="logging:\n  level: DEBUG\n")
    def test_load_yaml(self, mock_file):
        config = ConfigManager.load_yaml('dummy_path.yaml')
        self.assertEqual(config['logging']['level'], 'DEBUG')

        mock_file.assert_called_once_with('dummy_path.yaml', 'r')

    @patch('builtins.open', new_callable=mock_open, read_data="logging:\n  level: DEBUG\n")
    def test_load_yaml_file_not_found(self, mock_file):
        ConfigManager.setup_logger()
        mock_file.side_effect = FileNotFoundError

        with self.assertRaises(FileNotFoundError):
            ConfigManager.load_yaml('non_existent.yaml')

    @patch('builtins.open', new_callable=mock_open, read_data=": invalid yaml")
    def test_load_yaml_invalid_yaml(self, mock_file):
        ConfigManager.setup_logger()
        mock_file.side_effect = yaml.YAMLError

        with self.assertRaises(yaml.YAMLError):
            ConfigManager.load_yaml('invalid.yaml')

    def test_merge_config_with_cli_params_taking_priority(self):
        yaml_config = {
            'synthesis_parameters': {
                'max_depth': 10
            },
            'logging': {
                'level': 'DEBUG'
            }
        }

        cli_args = argparse.Namespace(synthesis_parameters__max_depth=15, logging_level=None)

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        self.assertEqual(merged_options.synthesis_parameters.max_depth, 15)
        self.assertEqual(merged_options.logging.level, 'DEBUG')

    def test_merge_config_with_yaml_taking_priority(self):
        yaml_config = {
            'synthesis_parameters': {
                'max_depth': 20
            },
            'logging': {
                'level': 'INFO'
            }
        }

        cli_args = argparse.Namespace(synthesisparameters__max_depth=None, logging__level=None)

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        self.assertEqual(merged_options.synthesis_parameters.max_depth, 20)
        self.assertEqual(merged_options.logging.level, 'INFO')

    def test_merge_config_with_default_taking_priority(self):
        yaml_config = {
            'synthesis_parameters': {
                'max_depth': None
            },
            'logging': {
                'level': None
            }
        }

        cli_args = argparse.Namespace(synthesisparameters__max_depth=None, logging__level=None)

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        self.assertEqual(merged_options.synthesis_parameters.max_depth, self.default_options.synthesis_parameters.max_depth)
        self.assertEqual(merged_options.logging.level, self.default_options.logging.level)


    @patch('src.cegis.z3.config_manager.ConfigManager.load_yaml', return_value=None)
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None))
    def test_get_config(self, mock_parse_args, mock_load_yaml):
        config = ConfigManager.get_config()
        self.assertIsInstance(config, Options)
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.cegis.z3.config_manager.ConfigManager.load_yaml', return_value={'logging': {'level': 'DEBUG'}})
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None, logging__level=None))
    def test_get_config_with_yaml(self, mock_parse_args, mock_load_yaml):
        config = ConfigManager.get_config()
        self.assertEqual(config.logging.level, 'DEBUG')
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.cegis.z3.config_manager.ConfigManager.load_yaml', return_value={'logging': {'level': 'DEBUG'}})
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None, logging__level='INFO'))
    def test_get_config_with_cli(self, mock_parse_args, mock_load_yaml):
        ConfigManager.setup_logger()
        config = ConfigManager.get_config()
        self.assertEqual(config.logging.level, 'INFO')
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.cegis.z3.config_manager.ConfigManager.load_yaml', side_effect=FileNotFoundError)
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None))
    def test_get_config_file_not_found(self, mock_parse_args, mock_load_yaml):
        ConfigManager.setup_logger()
        config = ConfigManager.get_config()
        self.assertIsInstance(config, Options)
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

if __name__ == '__main__':
    unittest.main()
