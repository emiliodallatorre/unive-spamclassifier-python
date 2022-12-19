import os
from math import log

import pandas


def read_data() -> pandas.DataFrame:
    data: pandas.DataFrame = pandas.read_csv("data/spambase.data")
    return data


def decorate_dataframe(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    dataframe.columns = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "spam",
    ]
    return dataframe


def get_raw_data() -> pandas.DataFrame:
    data: pandas.DataFrame = read_data()
    data: pandas.DataFrame = decorate_dataframe(data)
    return data


def calculate_weights(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    Calculates the weight of every word wit tf-idf.
    :param dataframe:
    :return:
    """

    weights: pandas.DataFrame = dataframe.copy()
    weights = weights.drop(columns=[
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "spam",
    ])

    # Calculate the tf-idf
    total_docs: int = len(dataframe)
    weights = weights.apply(lambda column: column * log(total_docs / (column > 0).sum()))

    # Add the spam column
    weights["spam"] = dataframe["spam"]

    return weights


def get_data() -> pandas.DataFrame:
    if not os.path.exists("data/weights.csv"):
        data: pandas.DataFrame = get_raw_data()
        data: pandas.DataFrame = calculate_weights(data)

        data.to_pickle("data/weights.pkl")
    else:
        data: pandas.DataFrame = pandas.read_pickle("data/weights.pkl")

    return data
