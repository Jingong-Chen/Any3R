# Open3D's point-cloud routines are multi-threaded and only bitwise reproducible single-threaded,
# which is how evaluate.py runs them; the parity tests compare to the last bit.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
