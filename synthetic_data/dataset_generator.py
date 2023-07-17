import copy
import json

from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator
import dataprofiler as dp

from synthetic_data.distinct_generators.categorical_generator import random_categorical
from synthetic_data.distinct_generators.datetime_generator import random_datetimes
from synthetic_data.distinct_generators.float_generator import random_floats
from synthetic_data.distinct_generators.int_generator import random_integers
from synthetic_data.distinct_generators.text_generator import random_string, random_text


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_data_to_df(
    np_data: np.array,
    path: Optional[str] = None,
    index: bool = False,
    column_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Converts np array to a pandas dataframe

    :param np_data: np array to be converted
    :type np_data: numpy array
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional
    :param index: whether to include index in output to csv
    :type path: bool, optional
    :param column_names: The names of the columns of a dataset
    :type path: List, None, optional
    :return: a pandas dataframe
    """
    # convert array into dataframe
    if not column_names:
        column_names = [x for x in range(len(np_data))]
    dataframe = pd.DataFrame.from_dict(dict(zip(column_names, np_data)))
    # save the dataframe as a csv file
    if path:
        dataframe.to_csv(path, index=index, encoding="utf-8")
        print(f"Created {path}!")
    return dataframe


def get_ordered_column(data: np.array, type: str) -> np.array:
    """
    Sorts a numpy array based on data type

    :param data: numpy array to be sorted
    :type data: np.array

    :return: sorted numpy array
    """
    if type == "datetime":
        original_format = "%B %d %Y %H:%M:%S"
        datetime = np.array([np.datetime64(dt, format=original_format) for dt in data])
        sorted_datetime64 = np.sort(datetime)
        sorted_datetime = [dt.strftime(original_format) for dt in sorted_datetime64]
        return sorted_datetime

    else:
        return np.sort(data)

def generate_dataset_by_class(
    report,
    rng: Generator,
    columns_to_generate: Optional[List[dict]] = None,
    dataset_length: int = 100000,
    path: Optional[str] = None,
    ordered: bool = False,
) -> pd.DataFrame:
    """
    Randomly generate a dataset with a mixture of different data classes

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param columns_to_generate: Classes of data to be included in the dataset
    :type columns_to_generate: List[dict], None, optional
    :param dataset_length: length of the dataset generated
    :type dataset_length: int, optional
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional
    :param ordered: whether to generate ordered data
    :type ordered: bool, optional

    :return: pandas DataFrame
    """
    gen_funcs = {
        "integer": random_integers,
        "float": random_floats,
        "categorical": random_categorical,
        "ordered": get_ordered_column,
        "text": random_text,
        "datetime": random_datetimes,
        "string": random_string,
    }

    if columns_to_generate is None:
        columns_to_generate = [
            dict(generator="datetime"),
            dict(generator="integer"),
            dict(generator="float"),
            dict(generator="categorical"),
            dict(generator="ordered"),
            dict(generator="text"),
            dict(generator="string"),
        ]

    dataset = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        col_generator = col_.pop("generator")
        if col_generator not in gen_funcs:
            raise ValueError(f"generator: {col_generator} is not a valid generator.")
        col_generator_function = gen_funcs.get(col_generator)
        if ordered:
            data_type = report["data_stats"][col].get("data_type", None)
            dataset.append(get_ordered_column(col_generator_function(**col_, num_rows=dataset_length, rng=rng), data_type))
        else:
            dataset.append(col_generator_function(**col_, num_rows=dataset_length, rng=rng))
    return convert_data_to_df(dataset, path)