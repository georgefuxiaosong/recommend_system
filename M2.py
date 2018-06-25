# -*- coding: utf-8 -*-
"""


@author: shentanyue
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
user_names = ['user_id', 'gender', 'age', 'occupation', 'zip'] #用户表的数据字段名
users = pd.read_table('C:\\Users\\Fuxiao\\Desktop\\待选数据集\\ml-1m\\users.dat', sep='::', header=None, names=user_names)
print(len(users))
users.head()