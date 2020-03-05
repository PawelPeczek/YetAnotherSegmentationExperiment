from typing import Dict, TypeVar, List, Tuple

K = TypeVar("K")
T = TypeVar("T")


def append_to_dictionary_of_lists(dictionary: Dict[K, List[T]],
                                  to_append: Tuple[K, List[T]]
                                  ) -> Dict[K, List[T]]:
    key, value = to_append
    if key in dictionary:
        dictionary[key].extend(value)
    else:
        dictionary[key] = value
    return dictionary
