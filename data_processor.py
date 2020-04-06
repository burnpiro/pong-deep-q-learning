from __future__ import absolute_import, division, print_function, unicode_literals
import functools
import pandas as pd

pd.set_option("display.precision", 2)

columns = ['mean', 'std', 'min_length', 'max_length', 'num_of_leafs', 'time']


def process_one_type(dataframes, name):
    rows = {
        'mean': [],
        'std': [],
        'min_length': [],
        'max_length': [],
        'num_of_leafs': [],
        'time': [],
        'ratio': 0,
        'operations': [],
        'name': name
    }
    operations = 0
    for item in dataframes:
        rows['operations'].append(item.describe()['mean']['count'])
        for column in columns:
            if column == 'min_length':
                rows[column].append(item.describe()[column]['min'])
            elif column == 'max_length':
                rows[column].append(item.describe()[column]['max'])
            else:
                rows[column].append(item.describe()[column]['mean'])

    rows = pd.DataFrame(rows)
    return pd.DataFrame({
        'mean': [rows.describe()['mean']['mean']],
        'std': [rows.describe()['std']['mean']],
        'min': [rows.describe()['min_length']['min']],
        'max': [rows.describe()['max_length']['max']],
        'leafs': [rows.describe()['num_of_leafs']['mean']],
        'time': [rows.describe()['time']['mean']],
        'ratio': [0],
        'operations': [rows.describe()['operations']['mean']],
        'operations_std': [rows.describe()['operations']['std']],
        'operations_min': [rows.describe()['operations']['min']],
        'operations_max': [rows.describe()['operations']['max']],
        'name': [name]
    })
