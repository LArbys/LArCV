import numpy  as np
import root_numpy as rn
import pandas as pd

print pd.DataFrame(rn.root2array("taritree.root")).drop_duplicates().index.size
