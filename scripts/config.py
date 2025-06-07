# import os
# import yaml

# def load_config(config_file="config.yaml"):
#     with open(config_file, "r") as f:
#         cfg = yaml.safe_load(f)

#     # Convert relative paths to absolute paths
#     base_dir = os.path.dirname(os.path.abspath(config_file))
#     cfg["data"]["monthly_data_folder"] = os.path.abspath(os.path.join(base_dir, cfg["data"]["monthly_data_folder"]))
#     cfg["data"]["combined_output_csv"] = os.path.abspath(os.path.join(base_dir, cfg["data"]["combined_output_csv"]))
#     cfg["paths"]["results_folder"] = os.path.abspath(os.path.join(base_dir, cfg["paths"]["results_folder"]))
#     cfg["paths"]["models_folder"] = os.path.abspath(os.path.join(base_dir, cfg["paths"]["models_folder"]))
#     cfg["paths"]["combined_csv"] = os.path.abspath(os.path.join(base_dir, cfg["paths"]["combined_csv"]))

#     return cfg



import os
import yaml

def load_config(config_file="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    config_path = os.path.join(project_root, config_file)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert relevant paths to absolute
    for key, value in cfg.get("data", {}).items():
        if isinstance(value, str)   and not os.path.isabs(value):
            cfg["data"][key] = os.path.join(project_root, value)

    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str) and not os.path.isabs(value):
            cfg["paths"][key] = os.path.join(project_root, value)


    return cfg


