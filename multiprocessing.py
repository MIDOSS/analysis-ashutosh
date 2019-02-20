import os
import multiprocessing


def manage_queue(current, remaining, workers):
    if len(current) != 0:
        for task in current:
            if task.is_alive():
                continue
            else:
                try:
                    task.join()
                    current.remove(task)
                except RuntimeError as err:
                    if 'cannot join current thread' in err.args[0]:
                        continue
                    else:
                        raise
    if len(current) != workers and len(remaining) != 0:
        remaining[0].start()
        current.append(remaining[0])
        remaining.remove(remaining[0])
        time.sleep(1)
        manage_queue(current, remaining, workers)
    if len(current) == 0  and  len(remaining) == 0:
        return
    else:
        time.sleep(1)
        manage_queue(current, remaining, workers)

# instead of loop, write target function that takes iterable argument 
def target_function(i):
    # do something
    



if __name__ == '__main__':
    processes= []
    for i in [YOUR_LIST / RANGE]:
        p = multiprocessing.Process(target = target_function, args = (argument1, argument2, ..., i))
        processes.append(p)
    manage_queue([], processes, 2) # edit number of processes you want alive at once

# All loops will be finished before next lines of code run becuase of process.join()

# Additional lines of code