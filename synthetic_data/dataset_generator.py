"""Contains generator that returns collective df of requested distinct generators."""

import copy
<<<<<<< HEAD
import logging
=======
>>>>>>> origin/feature/simple-tabular-generator
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

from synthetic_data.distinct_generators.categorical_generator import random_categorical
from synthetic_data.distinct_generators.datetime_generator import random_datetimes
from synthetic_data.distinct_generators.float_generator import random_floats
from synthetic_data.distinct_generators.int_generator import random_integers
from synthetic_data.distinct_generators.text_generator import random_string, random_text


def convert_data_to_df(
    np_data: np.array,
    path: Optional[str] = None,
    index: bool = False,
    column_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert np array to a pandas dataframe.

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
<<<<<<< HEAD
=======

>>>>>>> origin/feature/simple-tabular-generator
    # save the dataframe as a csv file
    if path:
        dataframe.to_csv(path, index=index, encoding="utf-8")
    return dataframe


def get_ordered_column(
    data: np.array,
    data_type: str,
    order: str = "ascending",
) -> np.array:
    """Sort a numpy array based on data type.

    :param data: numpy array to be sorted
    :type data: np.array

    :return: sorted numpy array
    """
    if data_type == "datetime":
        sorted_data = np.array(sorted(data, key=lambda x: x[1]))
        sorted_data = sorted_data[:, 0]

    else:
        sorted_data = np.sort(data)

    if order == "descending":
        return sorted_data[::-1]
    return sorted_data


<<<<<<< HEAD
def generate_dataset(
    rng: Generator,
    columns_to_generate: List[dict],
    dataset_length: int = 100000,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Randomizes a dataset with a mixture of different data classes.
=======
def generate_dataset_by_class(
    rng: Generator,
    columns_to_generate: Optional[List[dict]] = None,
    dataset_length: int = 100000,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """Randomly generate a dataset with a mixture of different data classes.
>>>>>>> origin/feature/simple-tabular-generator

    :param rng: the np rng object used to generate random values
    :type rng: numpy Generator
    :param columns_to_generate: Classes of data to be included in the dataset
    :type columns_to_generate: List[dict], None, optional
<<<<<<< HEAD
    :param dataset_length: length of the dataset generated, default 100,000
    :type dataset_length: int, optional
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional
=======
    :param dataset_length: length of the dataset generated
    :type dataset_length: int, optional
    :param path: path to output a csv of the dataframe generated
    :type path: str, None, optional
    :param ordered: whether to generate ordered data
    :type ordered: bool, optional
>>>>>>> origin/feature/simple-tabular-generator

    :return: pandas DataFrame
    """
    gen_funcs = {
        "integer": random_integers,
        "float": random_floats,
        "categorical": random_categorical,
        "text": random_text,
        "datetime": random_datetimes,
        "string": random_string,
    }

<<<<<<< HEAD
    if not columns_to_generate:
        logging.warning(
            "columns_to_generate is empty, empty dataframe will be returned."
        )
        return pd.DataFrame()

    dataset = []
    column_names = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        col_generator = col_.pop("generator")
        if col_generator not in gen_funcs:
            raise ValueError(f"generator: {col_generator} is not a valid generator.")
        if "name" in col_:
            name = col_.pop("name")
        else:
            name = col_generator
        col_generator_function = gen_funcs.get(col_generator)
=======
    dataset = []
    for col in columns_to_generate:
        col_ = copy.deepcopy(col)
        data_type_var = col_.get("data_type", None)
        if data_type_var not in gen_funcs:
            raise ValueError(f"generator: {data_type_var} is not a valid generator.")

        col_generator_function = gen_funcs.get(data_type_var)
>>>>>>> origin/feature/simple-tabular-generator
        generated_data = col_generator_function(
            **col_, num_rows=dataset_length, rng=rng
        )
        sort = col_.get("ordered", None)

        if sort in ["ascending", "descending"]:
            dataset.append(
                get_ordered_column(
                    generated_data,
<<<<<<< HEAD
                    col_generator,
=======
                    data_type_var,
>>>>>>> origin/feature/simple-tabular-generator
                    sort,
                )
            )
        else:
<<<<<<< HEAD
            if col_generator == "datetime":
=======
            if data_type_var == "datetime":
>>>>>>> origin/feature/simple-tabular-generator
                date = generated_data[:, 0]
                dataset.append(date)
            else:
                dataset.append(generated_data)
<<<<<<< HEAD
        column_names.append(name)
    return convert_data_to_df(dataset, path, column_names=column_names)
=======
    return convert_data_to_df(dataset, path)
>>>>>>> origin/feature/simple-tabular-generator
