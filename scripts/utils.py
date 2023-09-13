

import functools
import ray
import math
import numpy as np


class Argumentor:

  r"""Argumenting context manager.
  This allows a user to specify a set of arguments which will
  be partially applied to each of the callables passed to this
  context manager.
  With this context, we can ensure that each callabe uses the
  same set of arguments and simplify the code::
    add, sub = lambda x, y: x + y, lambda x, y: x - y
    # 15, -5
    with Argumentor([add, sub], y=10) as (add10, sub10):
      print("%d, %d" % (add10(5), sub10(5)))
  Arguments:
    function_or_functions (callable or iterable of callables):
      The target callable objects. These will be partially
      applied with the passed *args and **kwargs.
  """

  def __init__(self, function_or_functions, *args, **kwargs):

    apply = functools.partial

    try:
      functions = iter(function_or_functions)
    except:
      self.function_apps = [
        apply(function_or_functions, *args, **kwargs)]
    else:
      self.function_apps = [
        apply(function, *args, **kwargs)
          for function in functions]

  def __enter__(self):
    try:
      [function] = self.function_apps
    except:
      return self
    else:
      return function

  def __iter__(self):
    return iter(self.function_apps)

  def __exit__(self, type, value, traceback):
    ...


def chunk(l, n):
  u"""Partition `x` into `n` equal-sized (apporx.) bathes.
  """
  d, r = divmod(len(l), n)
  for i in range(n):
    si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
    yield l[si:si+(d+1 if i < r else d)]


def map_and_reduce(func):
  u"""Map and reduce for parallel execution.
  
  This is based on ray's multiprocessing framework, but
  over-parallelization is avoided with batching.
  
  """

  def main(tuples, n):

    ray.init()

    @ray.remote
    def process_batch(batch):
      batch_return = []
      for args in batch:
        batch_return.append(func(*args))
      return batch_return

    # map
    futures = []
    for batch in chunk(tuples, n):
      futures.append(process_batch.remote(batch))
    # reduce
    results = ray.get(futures)

    ray.shutdown()
    return results
  
  return main