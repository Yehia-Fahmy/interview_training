import numpy as np
from starter_02 import ClassificationMetrics

# Simple test case
y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 0])

metrics = ClassificationMetrics(y_true, y_pred)
conf_matrix = metrics.confusion_matrix()
precision = metrics.precision_per_class()

print("Confusion Matrix:")
print(conf_matrix)
print("\nPrecision per class:")
print(precision)
print("\nAccuracy:", metrics.accuracy())

# Expected:
# Confusion Matrix:
# [[2, 1, 0],   # Row 0: true class 0 -> predicted as [0,1,2]
#  [0, 3, 0],   # Row 1: true class 1 -> predicted as [0,1,2]
#  [1, 0, 2]]   # Row 2: true class 2 -> predicted as [0,1,2]
#
# Precision for class 0: TP=2, TP+FP=2+0+1=3 -> 2/3 = 0.667
# Precision for class 1: TP=3, TP+FP=1+3+0=4 -> 3/4 = 0.75
# Precision for class 2: TP=2, TP+FP=0+0+2=2 -> 2/2 = 1.0

