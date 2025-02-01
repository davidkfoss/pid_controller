import configparser


def load_config(filename="config.ini"):
    """Loads and parses the configuration file."""
    config = configparser.ConfigParser()
    config.read(filename)

    parsed_config = {}

    for section in config.sections():
        parsed_config[section] = {
            key: float(value) if "," not in value else tuple(
                map(float, value.split(",")))
            for key, value in config[section].items()
        }

    plant_type = parsed_config["General"]["plant"]
    controller_type = parsed_config["General"]["controller"]
    disturbance_params = parsed_config.get("Noise", {})

    return plant_type, controller_type, disturbance_params, parsed_config
