import json
import logging
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import argparse
import yaml
from src.utilities.options import Options
from src.utilities.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.default_options = Options()

    def test_generate_argparse_from_options(self):
        parser = ConfigManager.generate_argparse_from_options()
        self.assertIsInstance(parser, argparse.ArgumentParser)

        args = parser.parse_args(['--synthesis_parameters__max_depth', '6'])
        self.assertEqual(args.synthesis_parameters__max_depth, 6)

        args = parser.parse_args(['--logging__level', 'INFO'])
        self.assertEqual(args.logging__level, 'INFO')

    def test_setup_logger(self):
        ConfigManager.setup_logger()

        self.assertIsNotNone(ConfigManager.logger)

        self.assertEqual(ConfigManager.logger.level, logging.DEBUG)
        self.assertTrue(ConfigManager.logger.hasHandlers())
        self.assertEqual(len(ConfigManager.logger.handlers), 2)

        self.assertIsInstance(ConfigManager.logger.handlers[0], logging.FileHandler)
        self.assertIsInstance(ConfigManager.logger.handlers[1], logging.StreamHandler)

        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = project_root / "logs"
        self.assertTrue(log_dir.exists())

        log_files = list(log_dir.glob("config_*.log"))
        self.assertGreater(len(log_files), 0)

    def test_get_logger(self):
        logger1 = ConfigManager.get_logger()
        logger2 = ConfigManager.get_logger()

        self.assertIs(logger1, logger2)
        self.assertIs(logger1, ConfigManager.logger)


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

        self.assertEqual(merged_options.synthesis_parameters.max_depth,
                         self.default_options.synthesis_parameters.max_depth)
        self.assertEqual(merged_options.logging.level, self.default_options.logging.level)

    @patch('src.utilities.config_manager.ConfigManager.load_yaml', return_value=None)
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None))
    def test_get_config(self, mock_parse_args, mock_load_yaml):
        config = ConfigManager.get_config()
        self.assertIsInstance(config, Options)
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.utilities.config_manager.ConfigManager.load_yaml', return_value={'logging': {'level': 'DEBUG'}})
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None, logging__level=None))
    def test_get_config_with_yaml(self, mock_parse_args, mock_load_yaml):
        config = ConfigManager.get_config()
        self.assertEqual(config.logging.level, 'DEBUG')
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.utilities.config_manager.ConfigManager.load_yaml', return_value={'logging': {'level': 'DEBUG'}})
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None, logging__level='INFO'))
    def test_get_config_with_cli(self, mock_parse_args, mock_load_yaml):
        ConfigManager.setup_logger()
        config = ConfigManager.get_config()
        self.assertEqual(config.logging.level, 'INFO')
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    @patch('src.utilities.config_manager.ConfigManager.load_yaml', side_effect=FileNotFoundError)
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None))
    def test_get_config_file_not_found(self, mock_parse_args, mock_load_yaml):
        ConfigManager.setup_logger()
        config = ConfigManager.get_config()
        self.assertIsInstance(config, Options)
        mock_parse_args.assert_called_once()
        mock_load_yaml.assert_called_once()

    def test_merge_config_with_custom_grammar(self):
        yaml_config = {
            'synthesis_parameters': {
                'custom_grammar': '{"S": ["T", ["+", "S", "S"]], "T": ["x", "y", "1", "2"]}'
            }
        }

        cli_args = argparse.Namespace()

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        expected_grammar = {"S": ["T", ("+", "S", "S")], "T": ["x", "y", "1", "2"]}
        self.assertEqual(merged_options.synthesis_parameters.custom_grammar, expected_grammar)

    def test_merge_config_with_weighted_generator(self):
        yaml_config = {
            'synthesis_parameters': {
                'use_weighted_generator': True
            }
        }

        cli_args = argparse.Namespace()

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        self.assertTrue(merged_options.synthesis_parameters.use_weighted_generator)

    @patch('src.utilities.config_manager.ConfigManager.load_yaml', return_value={
        'synthesis_parameters': {
            'custom_grammar': '{"S": ["T", ["+", "S", "S"]], "T": ["x", "y", "1", "2"]}',
            'use_weighted_generator': True
        }
    })
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(config=None))
    def test_get_config_with_new_options(self, mock_parse_args, mock_load_yaml):
        config = ConfigManager.get_config()
        self.assertEqual(config.synthesis_parameters.custom_grammar,
                         {"S": ["T", ("+", "S", "S")], "T": ["x", "y", "1", "2"]})
        self.assertTrue(config.synthesis_parameters.use_weighted_generator)

    def test_merge_config_with_cli_custom_grammar(self):
        yaml_config = {}
        cli_args = argparse.Namespace(
            synthesis_parameters__custom_grammar='{"S": ["T", ["+", "S", "S"]], "T": ["x", "y", "1", "2"]}'
        )

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        expected_grammar = {"S": ["T", ("+", "S", "S")], "T": ["x", "y", "1", "2"]}
        self.assertEqual(merged_options.synthesis_parameters.custom_grammar, expected_grammar)

    def test_merge_config_with_cli_weighted_generator(self):
        yaml_config = {}
        cli_args = argparse.Namespace(synthesis_parameters__use_weighted_generator=True)

        merged_options = ConfigManager.merge_config(self.default_options, yaml_config, cli_args)

        self.assertTrue(merged_options.synthesis_parameters.use_weighted_generator)

    def test_generate_argparse_includes_new_options(self):
        parser = ConfigManager.generate_argparse_from_options()
        args = parser.parse_args(['--synthesis_parameters__custom_grammar',
                                  '{"S": ["T", ["+", "S", "S"]], "T": ["x", "y", "1", "2"]}',
                                  '--synthesis_parameters__use_weighted_generator'])

        self.assertEqual(json.loads(args.synthesis_parameters__custom_grammar),
                         {"S": ["T", ["+", "S", "S"]], "T": ["x", "y", "1", "2"]})
        self.assertTrue(args.synthesis_parameters__use_weighted_generator)


if __name__ == '__main__':
    unittest.main()
