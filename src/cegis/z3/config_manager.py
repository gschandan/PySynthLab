import argparse
import json
from dataclasses import asdict, fields
from typing import Dict, Any, Optional
import yaml

from src.cegis.z3.options import Options, LoggingOptions, SynthesisParameters, SolverOptions


class ConfigManager:
    @staticmethod
    def generate_argparse_from_options() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="PySynthLab Synthesiser")
        parser.add_argument('--config', type=str, help="Path to custom config file")
        parser.add_argument('input_source', nargs='?', default='stdin',
                            help="Source of the input problem (stdin or file path)")

        for class_name in ['LoggingOptions', 'SynthesisParameters', 'SolverOptions']:
            class_obj = globals()[class_name]
            prefix = class_name.lower().replace('options', '')
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
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
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
                if key in merged_dict and isinstance(merged_dict[key], dict):
                    merged_dict[key].update(value)
                else:
                    merged_dict[key] = value

        for arg_name, arg_value in vars(cli_args).items():
            if arg_value is not None:
                parts = arg_name.split('__', 1) 
                if len(parts) > 1 and parts[0] in merged_dict:
                    merged_dict[parts[0]][parts[1]] = arg_value
                else:
                    merged_dict[arg_name] = arg_value

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
        default_options = Options()
        parser = ConfigManager.generate_argparse_from_options()
        args = parser.parse_args()

        yaml_config = None
        if args.config:
            yaml_config = ConfigManager.load_yaml(args.config)
        else:
            try:
                yaml_config = ConfigManager.load_yaml("../config/user_config.yaml")
            except FileNotFoundError:
                pass

        return ConfigManager.merge_config(default_options, yaml_config, args)
