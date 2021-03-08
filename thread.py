import concurrent.futures

with concurrent.futures.ProcessPoolExecutor() as executor:

    def _thread(i):
        # margins = _ensemble[i].predict(d_test, output_margin=True)
        # return "test"
        return "test"

    with concurrent.futures.ThreadPoolExecutor(max_workers = 100) as executor:
        futures = []
        for i in range(10 - 1):
            future = executor.submit(_thread, i)
            futures.append(future)
        finished = concurrent.futures.wait(futures, 2)
        print(finished)