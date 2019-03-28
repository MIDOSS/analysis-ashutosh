import time
import multiprocessing

def manage_queue(remaining, workers, current=[]):
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
        manage_queue(remaining, workers, current)
    if len(current) == 0  and  len(remaining) == 0:
        return
    else:
        time.sleep(1) # may want to increment this if each process takes over 40 minutes to complete
        manage_queue(remaining, workers, current)

# instead of loop, write target function that takes iterable argument 
def target_function(i):
    # do something

num_processes_alive = 4
if __name__ == '__main__':
    processes= []
    for i in []:
        p = multiprocessing.Process(target = target_function, args = (i, *kwargs))
        processes.append(p)
    manage_queue(processes, num_processes_alive)
