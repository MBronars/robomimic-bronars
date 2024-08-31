import time

def tic():
    return time.perf_counter()

def toc(start_time):
    end_time = time.perf_counter()
    return end_time - start_time