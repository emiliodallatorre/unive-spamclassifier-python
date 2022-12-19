import time


def chrono_function(func, *args, **kwargs) -> tuple:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return result, end_time - start_time
