import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('checkpoint_v1.tpg', 'rb') as f:
    trainer_v1 = pickle.load(f)
with open('checkpoint_v2.tpg', 'rb') as f:
    trainer_v2 = pickle.load(f)
with open('checkpoint_v3.tpg', 'rb') as f:
    trainer_v3 = pickle.load(f)
with open('checkpoint_v4.tpg', 'rb') as f:
    trainer_v4 = pickle.load(f)
with open('checkpoint_v5.tpg', 'rb') as f:
    trainer_v5 = pickle.load(f)

v1_x = np.asarray([x[0] for x in trainer_v1['results']])
v1_y = np.asarray([x[1] for x in trainer_v1['results']])

v2_x = np.asarray([x[0] for x in trainer_v2['results']])
v2_y = np.asarray([x[1] for x in trainer_v2['results']])

v3_x = np.asarray([x[0] for x in trainer_v3['results']])
v3_y = np.asarray([x[1] for x in trainer_v3['results']])

v4_x = np.asarray([x[0] for x in trainer_v4['results']])
v4_y = np.asarray([x[1] for x in trainer_v4['results']])

v5_x = np.asarray([x[0] for x in trainer_v5['results']])
v5_y = np.asarray([x[1] for x in trainer_v5['results']])

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(v1_x, v1_y, label='Version 1')  # Plot some data on the axes.
ax.plot(v2_x, v2_y, label='Version 2')  # Plot more data on the axes...
ax.plot(v3_x, v3_y, label='Version 3')  # ... and some more.
ax.plot(v4_x, v4_y, label='Version 4')  # ... and some more.
ax.plot(v5_x, v5_y, label='Version 5')  # ... and some more.
ax.set_xlabel('Generation')  # Add an x-label to the axes.
ax.set_ylabel('Fitness')  # Add a y-label to the axes.
ax.set_title("TPG Version Fitness")  # Add a title to the axes.
ax.legend()  # Add a legend.

fig.savefig('temp.png')