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
                if "," in value:
                    parsed_config[section][key] = [v.strip()
                                                   for v in value.split(",")]
                else:
                    parsed_config[section][key] = value

    plant_type = parsed_config["General"]["plant"]
    controller_type = parsed_config["General"]["controller"]
    disturbance_params = parsed_config.get("Noise", {})
    training_epochs = int(parsed_config["General"]["training_epochs"])
    timesteps_per_epoch = int(parsed_config["General"]["timesteps_per_epoch"])

    return plant_type, controller_type, disturbance_params, training_epochs, timesteps_per_epoch, parsed_config
