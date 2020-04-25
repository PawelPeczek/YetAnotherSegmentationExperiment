import json
import os
from typing import Optional, Union

from src.primitives.files import ParsedJSON


def safe_parse_json(path: str) -> Optional[ParsedJSON]:
    try:
        return parse_json(path=path)
    except Exception:
        return None


def parse_json(path: str) -> ParsedJSON:
    with open(path, "r") as file:
        return json.load(fp=file)


def dump_json(path: str, content: Union[list, dict]) -> None:
    prepare_file_parent_dir(file_path=path)
    with open(path, "w") as file:
        return json.dump(content, file, indent=4)


def extract_file_name_without_extension(path: str) -> str:
    file_name = os.path.basename(path)
    return file_name.split(sep=".")[0]


def make_path_relative(path: str, reference_path: str) -> Optional[str]:
    if not path.startswith(reference_path):
        return None
    relative_path = path[len(reference_path):]
    if not relative_path.startswith('/'):
        return None
    return relative_path[1:]


def prepare_file_parent_dir(file_path: str) -> None:
    file_path = os.path.abspath(file_path)
    prepare_storage_dir(path=os.path.dirname(file_path))


def prepare_storage_dir(path: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)


def escape_base_name(base_name: str) -> str:
    base_name = base_name.lower()
    chars_to_discard = "%*'\"/\\"
    for char_to_discard in chars_to_discard:
        base_name = base_name.replace(char_to_discard, "")
    return base_name.replace(" ", "_")
