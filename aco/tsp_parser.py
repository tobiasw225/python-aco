#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import re
import os
import numpy as np


def read_tsp_data(file: str):
    """
        we open the TSP file and put each line cleaned of spaces
        and newline characters in a list

    :param file:
    :return:
    """
    with open(file) as f:
        content = f.read()
        data = re.findall(r'\d+ \d+\.\d+\s\d+.\d+', content)
        dim = detect_dimension(content)
        if data:
            return data, dim


def detect_dimension(content: str) -> int:
    """

    :param content:
    :return:
    """
    res = re.search(r'DIMENSION:\s*([\d]+)', content)
    if res:
        return int(res.group(1))
    else:
        raise TypeError('Wrong data format for tsp-problem.')


def get_cities(my_list: list,
               dim: int,
               num_cities: int = -1):
    """
        Iterate through the list of line from the file
        if the line starts with a numeric value within the
        range of the dimension , we keep the rest which are
        the coordinates of each city
        1 33.00 44.00 results to 33.00 44.00

    :param my_list:
    :param dim:
    :param num_cities:
    :return:
    """
    cities = {}

    for item in my_list:
        for num in range(1, dim + 1):
            position = item.partition(' ')[-1]
            x = np.array(position.split()).astype(float)
            if position not in cities:
                cities[position] = x

    return list(cities.values())[:num_cities]


def create_cities_dict(cities):
    """
        We zip for reference each city to a number
        in order to work and solve the TSP we need a list
        of cities like
        [1,2,3,4,5,...........]
        with the dictionary we will have a reference of the coordinates of each city
        to calculate the distance within (i + 1, i) or (2 - 1) were 2 and 1 represents each city

    :param cities:
    :return:
    """
    return dict(zip((range(1, len(cities) + 1)), cities))


def load_tsp_problem(file: str, num_cities: int) -> dict:
    """

    :param file:
    :return:
    """
    assert os.path.isfile(file)
    data, dimension = read_tsp_data(file)
    cities = get_cities(data, dimension, num_cities=num_cities)
    return create_cities_dict(cities)
