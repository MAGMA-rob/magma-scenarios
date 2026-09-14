"""Inspect installed scenario providers without registering environments."""

import argparse
import json


def main(command: str) -> int:
    parser = argparse.ArgumentParser(description="Inspect installed MAGMA scenarios.")
    if command == "show":
        parser.add_argument("scenario", help="Installed scenario ID")
    args = parser.parse_args()
    from . import get_scenario, list_scenarios

    if command == "list":
        print("\n".join(list_scenarios()))
    else:
        try:
            manifest = get_scenario(args.scenario)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps({
            "id": manifest.id,
            "definitions": dict(manifest.definitions),
            "presets": dict(manifest.presets),
            "skills": dict(manifest.skills),
            "environments": dict(manifest.environments),
        }, indent=2))
    return 0
