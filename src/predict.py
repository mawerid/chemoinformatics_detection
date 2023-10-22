import pandas as pd
import numpy as np


def predict(data: str) -> [float, float, float]:
    ic50 = np.random.ranf() * 10
    cc50 = np.random.ranf() * 10
    si = cc50 / ic50
    return ic50, cc50, si
