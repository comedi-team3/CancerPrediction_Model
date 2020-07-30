def batch_list(array, n):
    batches = []
    for i in range(0, len(array), n):
        batches.append(array[i:i + n])

    return batches

def acc(yhat, y):
    assert len(yhat) == len(y)
    return sum(1 for a, b in zip(yhat, y) if a == b) / float(len(y))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs