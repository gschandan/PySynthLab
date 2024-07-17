import argparse
import json
import logging
import os
import re
from dataclasses import asdict, fields
from typing import Dict, Any, Optional
import yaml
from src.cegis.z3.options import Options, LoggingOptions, SynthesisParameters, SolverOptions


class ConfigManager:
    logger = None

    def __init__(self):
        if ConfigManager.logger is None:
            ConfigManager.setup_logger()

    @staticmethod
    def setup_logger():
        if ConfigManager.logger is not None:
            return

        logger = logging.getLogger(__name__)

        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(os.path.join(log_dir, "default.log"))
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        ConfigManager.logger = logger

    @staticmethod
    def generate_argparse_from_options() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="PySynthLab Synthesiser")
        parser.add_argument('--config', type=str, help="Path to custom config file")
        parser.add_argument('input_source', nargs='?', default='stdin',
                            help="Source of the input problem (stdin or file path)")

        for class_name in ['LoggingOptions', 'SynthesisParameters', 'SolverOptions']:
            class_obj = globals()[class_name]
            prefix = ConfigManager.camel_to_snake(class_name).replace('_options', '')
            for field_obj in fields(class_obj):
                arg_name = f"--{prefix}__{field_obj.name}"
                kwargs = {
                    "dest": f"{prefix}__{field_obj.name}",
                    "help": field_obj.metadata.get('description', ''),
                    "default": argparse.SUPPRESS,
                }

                if field_obj.metadata.get('type') == 'bool':
                    kwargs['action'] = 'store_true' if not field_obj.default else 'store_false'
                    if field_obj.default:
                        arg_name = f"--no-{prefix}-{field_obj.name}"
                elif field_obj.name == 'operation_costs':
                    kwargs['type'] = lambda x: json.loads(x.replace("'", '"'))
                else:
                    kwargs['type'] = eval(field_obj.metadata.get('type', 'str'))
                    if 'choices' in field_obj.metadata:
                        kwargs['choices'] = field_obj.metadata['choices']
                        kwargs['help'] += f" Choices: {', '.join(map(str, field_obj.metadata['choices']))}"

                kwargs['help'] += f" (default: {field_obj.default})"
                parser.add_argument(arg_name, **kwargs)

        return parser

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            Dict[str, Any]: Configuration as a dictionary.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            ConfigManager.logger.error(f"Config file {file_path} not found")
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            ConfigManager.logger.error(f"Error parsing YAML file: {e}")
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    @staticmethod
    def merge_config(default_options: Options, yaml_config: Optional[Dict[str, Any]], cli_args: argparse.Namespace) \
            -> Options:
        """
        Merge default options with any user defined YAML config and CLI arguments.
        CLI arguments will take priority over any user defined config in the user_config.yaml file.

        Args:
            default_options (Options): Default options from the Options dataclass.
            yaml_config (Optional[Dict[str, Any]]): Configuration loaded from YAML file.
            cli_args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            Options: Merged configuration as Options dataclass.
        """
        merged_dict = asdict(default_options)

        if yaml_config:
            for key, value in yaml_config.items():
                if value is not None:
                    if isinstance(value, dict) and key in merged_dict:
                        for sub_key, sub_value in value.items():
                            if sub_value is not None:
                                merged_dict[key][sub_key] = sub_value
                    else:
                        merged_dict[key] = value

        for arg_name, arg_value in vars(cli_args).items():
            if arg_value is not None:
                parts = arg_name.split('__', 1)
                if len(parts) > 1 and parts[0] in merged_dict:
                    merged_dict[parts[0]][parts[1]] = arg_value
                else:
                    merged_dict[arg_name] = arg_value

        for field in fields(default_options):
            if isinstance(getattr(default_options, field.name), dict):
                for subfield in fields(getattr(default_options, field.name)):
                    if merged_dict[field.name].get(subfield.name) is None:
                        merged_dict[field.name][subfield.name] = getattr(getattr(default_options, field.name),
                                                                         subfield.name)
            elif merged_dict.get(field.name) is None:
                merged_dict[field.name] = getattr(default_options, field.name)

        return Options(
            logging=LoggingOptions(**merged_dict['logging']),
            synthesis_parameters=SynthesisParameters(**merged_dict['synthesis_parameters']),
            solver=SolverOptions(**merged_dict['solver']),
            input_source=merged_dict['input_source']
        )

    @staticmethod
    def get_config() -> Options:
        """
        Get the final configuration by merging defaults, YAML config, and CLI args.

        Returns:
            Options: Final merged configuration as Options dataclass.
        """
        ConfigManager.setup_logger()
        default_options = Options()
        parser = ConfigManager.generate_argparse_from_options()
        args = parser.parse_args()

        yaml_config = None
        if args.config:
            yaml_config = ConfigManager.load_yaml(args.config)
        else:
            try:
                yaml_config = ConfigManager.load_yaml("config/user_config.yaml")
            except FileNotFoundError:
                ConfigManager.logger.warning("Default config file '../config/user_config.yaml' not found. "
                                             "Using default options.")

        return ConfigManager.merge_config(default_options, yaml_config, args)

    @staticmethod
    def camel_to_snake(name):
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        return pattern.sub('_', name).lower()
