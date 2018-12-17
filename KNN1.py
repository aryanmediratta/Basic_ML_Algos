import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = { 'k':[[1,2],[2,3],[3,1]], 
			'r':[[5,7],[6,5],[7,6]]}

new_feature = [3,4]

def k_nearest_neighbors(data, predict, k=3):
	distances = []
	for group in data:
		for features in data[group]:
			euclid_distances = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclid_distances, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

result = k_nearest_neighbors(dataset, new_feature, k=3)
print(result)

for i in dataset:
	for j in dataset[i]:
		[[plt.scatter(j[0], j[1], color = i) for j in dataset[i]] for i in dataset]

plt.scatter(xs,ys)
plt.scatter(new_feature[0], new_feature[1], color=result)
plt.show()
