import argparse
import os
import subprocess

import yaml


def main(cfg, other_args):
    exp_config_file = os.path.join("exp_configs", f"{cfg.exp_config}.yaml")

    with open(exp_config_file, "r") as f:
        commands = yaml.safe_load(f)["commands"]

    for i, cs in enumerate(commands):
        c = " ".join(cs)
        c = f"{c} {other_args}"
        print(f"launching {c}")
        subprocess.run(c.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--exp_config", type=str, required=True)
    args, unknown_args = parser.parse_known_args()
    other_args = " ".join(unknown_args)
    main(args, other_args)