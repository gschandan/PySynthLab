import argparse
from dataclasses import asdict, fields
from typing import Dict, Any, Optional
import yaml

from src.cegis.z3.options import Options


class ConfigManager:
    @staticmethod
    def generate_argparse_from_options() -> argparse.ArgumentParser:
        """
        Generate an ArgumentParser based on the Options dataclass.

        Returns:
            argparse.ArgumentParser: Argument parser object.

        Example:
            >>> parser = ConfigManager.generate_argparse_from_options()
            >>> args = parser.parse_args(['--synthesis_parameters_max_depth', '6'])
            >>> print(args.synthesis_parameters_max_depth)
            6
        """
        parser = argparse.ArgumentParser(description="PySynthLab Synthesiser")
        parser.add_argument('--config', type=str, help="Path to custom config file")
        parser.add_argument('input_source', nargs='?', default='stdin',
                    help="Source of the input problem (stdin or file path)")

        for field_obj in fields(Options):
            arg_name = f"--{field_obj.name}"
            kwargs = {
                "dest": field_obj.name,
                "help": field_obj.metadata.get('description', ''),
                "default": argparse.SUPPRESS,
            }

            if field_obj.metadata.get('type') == 'bool':
                kwargs['action'] = 'store_true' if not field_obj.default else 'store_false'
                if field_obj.default:
                    arg_name = f"--no-{field_obj.name}"
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

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        if yaml_config:
            merged_dict = update_nested_dict(merged_dict, yaml_config)

        cli_dict = {}
        for field_obj in fields(Options):
            parts = field_obj.name.split('_')
            if len(parts) > 1:
                section = parts[0]
                key = '_'.join(parts[1:])
                if section not in cli_dict:
                    cli_dict[section] = {}
                if hasattr(cli_args, field_obj.name):
                    value = getattr(cli_args, field_obj.name)
                    if value is not None:
                        cli_dict[section][key] = value
            else:
                if hasattr(cli_args, field_obj.name):
                    value = getattr(cli_args, field_obj.name)
                    if value is not None:
                        cli_dict[field_obj.name] = value

        merged_dict = update_nested_dict(merged_dict, cli_dict)

        flat_dict = {}
        for section, values in merged_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_dict[f"{section}_{key}"] = value
            else:
                flat_dict[section] = values

        return Options(**flat_dict)

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

        yaml_config = ConfigManager.load_yaml("../config/user_config.yaml")
        if args.config:
            yaml_config = ConfigManager.load_yaml(args.config)

        return ConfigManager.merge_config(default_options, yaml_config, args)
