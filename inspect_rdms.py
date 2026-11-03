"""Stage 2: peek at the structure of the FOUNDCOG exemplar-level RDM pickles."""
import pickle, sys, numpy as np

def describe(obj, name="obj", depth=0, max_depth=4):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict with {len(obj)} keys, e.g. {list(obj.keys())[:6]}")
        if depth < max_depth:
            for k in list(obj.keys())[:3]:
                describe(obj[k], repr(k), depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} of length {len(obj)}")
        if depth < max_depth and len(obj):
            describe(obj[0], "[0]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} {str(obj)[:80]}")

path = sys.argv[1]
with open(path, "rb") as f:   # public dataset from the paper's authors
    data = pickle.load(f)
describe(data, path.split("/")[-1])
