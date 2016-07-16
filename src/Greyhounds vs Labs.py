import numpy as np 
import matplotlib.pyplot as plt
#This is used to show how important good features for your classifier truly are
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 28 + 4 * np.random.randn(greyhounds)
plt.hist([grey_height, lab_height], stacked = True, color =['r','b'])
plt.show()
#notice how in the middle the two dogs' probability of being the same height is almost the same. This is exactly why multiple features are useful
#never add useless or redundant features it will hurt the accuracy of your program
#make features easy to understand