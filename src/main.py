from dataclasses import asdict

from src.utilities.config_manager import ConfigManager


def main():
    config = ConfigManager.get_config()
    print(asdict(config))


if __name__ == "__main__":
    main()
