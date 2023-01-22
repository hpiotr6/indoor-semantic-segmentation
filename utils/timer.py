import contextlib
import os
import time


@contextlib.contextmanager
def timer(logger):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {end - start} seconds")
    os.makedirs(logger.log_dir) if not os.path.exists(logger.log_dir) else None
    with open(os.path.join(logger.log_dir, "times.txt"), "w") as f:
        f.write(f"train, {elapsed}")
