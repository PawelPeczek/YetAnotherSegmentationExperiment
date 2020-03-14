from typing import Optional


def safe_cast_str2int(number: str) -> Optional[int]:
    try:
        return int(number)
    except ValueError:
        return None
