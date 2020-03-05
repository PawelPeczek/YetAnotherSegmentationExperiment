import json
import os
from typing import Optional

from src.primitives.files import ParsedJSON


def safe_parse_json(path: str) -> Optional[ParsedJSON]:
    try:
        return parse_json(path=path)
    except Exception:
        return None


def parse_json(path: str) -> ParsedJSON:
    with open(path, "r") as file:
        return json.load(fp=file)


def extract_file_name_without_extension(path: str) -> str:
    file_name = os.path.basename(path)
    return file_name.split(sep=".")[0]
