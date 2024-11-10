# utilities

from itertools import chain
from pathlib import Path
import pandas as pd


def flatten_list(lists):
    return list(chain.from_iterable(lists))

# used concat instead of append because append is now deprecated in pandas
def append_to_csv(csv_filename, data):
    filename = Path(csv_filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(filename) if filename.exists() else pd.DataFrame()
    
    # for dict/data
    data_to_append = [data] if isinstance(data, dict) else data
    df = pd.concat([df, pd.DataFrame(data_to_append)], ignore_index=True)
    
    df.to_csv(filename, index=False)

