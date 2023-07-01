from itertools import repeat

def starmap_with_kwargs(pool, fn, kwargs_iter):
    args_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args,**kwargs)

def apply_kwargs(fn,kwargs):
    return fn(**kwargs)

def batch_list(data,samples_per_batch):
    data_batched = []
    batch_i = []
    for i, d in enumerate(data):
        batch_i.append(d)
        if len(batch_i) == samples_per_batch:
            data_batched.append(batch_i)
            batch_i = []
    
    if len(batch_i) >0:
        data_batched.append(batch_i)
    return data_batched
        