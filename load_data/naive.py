import sys
sys.path.append("..")
from utils.utils import load_raw_dataset

def load_data(
    path,
    pre_context = [],
):
    return load_raw_dataset(path, is_lines=True, has_line_index=False)