import multiprocessing as mp
from functools import partial
from geiger.util.progress import Progress

import numpy as np

def apply_func(func, queue, args_chunk):
    # Apply each group of arguments in a list of arg groups to a func.
    results = []
    for args in args_chunk:
        result = func(*args)
        results.append(result)

        # For progress.
        queue.put(1)

    return results

def parallelize(func, args_set, cpus=0):
    """
    Example:

    func:

        def func(a):
            return a

    args_set:

        [(1,),(2,),(3,),(4,)]
    """
    if cpus < 1:
        cpus = mp.cpu_count() - 1
    pool = mp.Pool(processes=cpus)
    print('Running on {0} cores.'.format(cpus))

    # Split args set into roughly equal-sized chunks, one for each core.
    args_chunks = np.array_split(args_set, cpus)
    print(args_chunks)

    # Create a queue so we can log everything to a single file.
    manager = mp.Manager()
    queue = manager.Queue()

    # A callback on completion.
    def done(results):
        queue.put(None)

    results = pool.map_async(partial(apply_func, func, queue), args_chunks, callback=done)

    # Print progress.
    p = Progress()
    comp = 0
    p.print_progress(comp/len(args_set))
    while True:
        msg = queue.get()
        p.print_progress(comp/len(args_set))
        if msg is None:
            break
        comp += msg

    # Flatten results.
    return [i for sub in results.get() for i in sub]
