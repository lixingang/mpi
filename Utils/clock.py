# descrbe.py
import time
import functools

def clock(func):
    """this is outer clock function"""

    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        print(func.__name__ + " func time_cost -> {}".format(time_cost))
        return result
    return clocked  # --> 3

class Timer():
    def __init__(self, name):
        self.start = time.time()
        self.name = name
    def end(self):
        print(f"[Timeit]{self.name}", time.time()-self.start)
    
# @functools.lru_cache()  # --> 5
# @clock  # --> 6
# def fib(n):
#     """this is fibonacci function"""
#     return n if n < 2 else fib(n - 1) + fib(n - 2)

if __name__ == "__main__":
    # 如果有 @functools.wraps(func)  # --> 4 大多数情况下我们希望的输出是这样的
    fib(1) # 输出 fib func time_cost -> 9.5367431640625e-07
    print(fib.__name__)  # 输出 fib
    print(fib.__doc__)  # 输出 this is fibonacci function
    
    # 如果没有@functools.wraps(func)  # --> 4
    fib(1) # 输出 fib func time_cost -> 9.5367431640625e-07
    print(fib.__name__)  # 输出 clocked
    print(fib.__doc__)  # 输出 this is inner clocked function