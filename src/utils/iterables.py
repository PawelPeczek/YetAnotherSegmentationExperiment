import multiprocessing
import operator
from functools import reduce
from typing import Dict, TypeVar, List, Tuple, Iterable, Callable, Optional, \
    Any, Set

import numpy as np

K = TypeVar("K")
V = TypeVar("V")
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


def split_dictionary_by_value(dictionary: Dict[K, Tuple[V, V]],
                              ) -> Tuple[Dict[K, V], Dict[K, V]]:
    left_dict, right_dict = {}, {}
    for key, value in dictionary:
        left_dict[key], right_dict[key] = value
    left_dict = remove_empty_values_from_dictionary(dictionary=left_dict)
    right_dict = remove_empty_values_from_dictionary(dictionary=right_dict)
    return left_dict, right_dict


def remove_empty_values_from_dictionary(dictionary: Dict[K, Optional[V]]
                                        ) -> Dict[K, V]:
    return {
        key: value for key, value in dictionary.items() if value is not None
    }


def count_dictionary_values(dictionary: Dict[K, List[Any]]) -> Dict[K, int]:
    return {key: len(value) for key, value in dictionary.items()}


def sum_dictionary_values(dictionary: Dict[Any, int]) -> int:
    return sum(dictionary.values())


def flatten_dictionary_of_lists(dictionary: Dict[K, List[V]],
                                keys_to_take: Optional[Set[K]] = None
                                ) -> List[V]:
    if keys_to_take is not None:
        dictionary = filter_dictionary_by_keys(
            dictionary=dictionary,
            keys_to_take=keys_to_take
        )
    return reduce(operator.add, dictionary.values(), [])


def filter_dictionary_by_keys(dictionary: Dict[K, Any],
                              keys_to_take: Set[K]
                              ) -> Dict[K, Any]:
    return {
        key: value
        for key, value in dictionary.items()
        if key in keys_to_take
    }


def add_grouping_to_dictionary(dictionary: Dict[K, List[V]],
                               group_by: Callable[[V], Optional[T]]
                               ) -> Dict[K, Dict[T, List[V]]]:
    return {
        key: group_list_by(value, group_by) for key, value in dictionary.items()
    }


def group_list_by(to_group: List[V],
                  group_by: Callable[[V], Optional[T]]
                  ) -> Dict[T, List[V]]:
    elements_and_classes = (
        (group_by(element), element) for element in to_group
    )
    elements_and_classes = (
        e for e in elements_and_classes if e[0] is not None
    )
    return reduce(append_to_dictionary_of_lists, elements_and_classes, {})


def extract_and_merge_dictionary_sub_groups(dictionary: Dict[K, Dict[T, List[V]]],
                                            to_extract: Dict[K, Set[T]]
                                            ) -> Dict[K, List[V]]:
    return {
        key: flatten_dictionary_of_lists(value, keys_to_take=to_extract[key])
        for key, value in dictionary.items()
    }


def random_sample(to_sample: List[Any]) -> Any:
    index = int(round(np.random.uniform() * (len(to_sample) - 1)))
    return to_sample[index]


def parallel_map(iterable: List[V],
                 map_function: Callable[[V], T],
                 workers_number: int
                 ) -> List[T]:
    pool = multiprocessing.Pool(processes=workers_number)
    result = pool.map(map_function, iterable)
    pool.join()
    return result


def split_list(to_split: List[Tuple[K, T]]) -> Tuple[List[K], List[T]]:

    def reducer(accumulator: Tuple[List[K], List[T]],
                element: Tuple[K, T]
                ) -> Tuple[List[K], List[T]]:
        left_list, right_list = accumulator
        left_element, right_element = element
        left_list.append(left_element)
        right_list.append(right_element)
        return left_list, right_list

    return reduce(reducer, to_split, ([], []))
