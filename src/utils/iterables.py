from typing import Dict, TypeVar, List, Tuple, Iterable, Callable

K = TypeVar("K")
T = TypeVar("T")


def append_to_dictionary_of_lists(dictionary: Dict[K, List[T]],
                                  to_append: Tuple[K, T]
                                  ) -> Dict[K, List[T]]:
    key, value = to_append
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return dictionary


def for_each(iterable: Iterable[T], side_effect: Callable[[T], None]) -> None:
    for element in iterable:
        side_effect(element)
