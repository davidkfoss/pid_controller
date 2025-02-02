import configparser


def load_config(filename="config.ini"):
    """Loads and parses the configuration file."""
    config = configparser.ConfigParser()
    config.read(filename)

    parsed_config = {}

    for section in config.sections():
        parsed_config[section] = {}

        for key, value in config[section].items():
            # Try to convert to float if possible
            if ";" in value:
                value = value.split(";")[0]  # Remove comments
            try:
                if "," in value:  # Handle tuples for comma-separated values
                    parsed_config[section][key] = tuple(
                        map(float, value.split(",")))
                else:
                    parsed_config[section][key] = float(value)
            except ValueError:
                # Keep as string if conversion fails
                parsed_config[section][key] = value

    plant_type = parsed_config["General"]["plant"]
    controller_type = parsed_config["General"]["controller"]
    disturbance_params = parsed_config.get("Noise", {})

    return plant_type, controller_type, disturbance_params, parsed_config
