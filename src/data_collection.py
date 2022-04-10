import pandas as pd
import datasets
from collections import Counter


def get_data(dataset_name="ucberkeley-dlab/measuring-hate-speech", columns=["text", "hatespeech"]):
    """
    Helper method which fetches the requested dataset, narrows it down to the
    relevant columns, aggregates second column to the most frequent value
    based on the first column, and returns it

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to be downloaded. For this project, the default
        value is "ucberkeley-dlab/measuring-hate-speech".
    columns : list, optional
        A list of columns to be extracted. For this project, the default value
        is  ["text", "hatespeech"].

    Returns
    -------
    data : pandas.DataFrame
        The fetched and processed dataset.

    """
    print("Fetching data...")
    dataset = datasets.load_dataset(dataset_name, "binary")
    data = dataset["train"].to_pandas()[columns]

    print("Processing...")
    data[columns[1]] = pd.to_numeric(
        data[columns[1]],
        downcast="integer"
    )

    data.loc[data[columns[1]] == 2, columns[1]] = 1

    data = data.groupby(columns[0]).agg(
        lambda x: Counter(x).most_common(1)[0][0]
    ).reset_index()

    print("Done!")
    return data