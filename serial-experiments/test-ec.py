import pickle, random
import dill

data = []
for i in range(1000):
    data.append(random.randint(0, 255))

pickled_data = pickle.dumps(data, 5)

for i in range(len(pickled_data)):
    if random.randint(0, 1000) == 75:
        pickled_data = pickled_data[:i] + b"\x80" + pickled_data[i + 1 :]

dill.loads(pickled_data)

data_out = pickle.loads(pickled_data)

for i in range(1000):
    assert data[i] == data_out[i]
