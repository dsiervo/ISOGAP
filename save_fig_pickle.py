# script to load a pickle matplot lib figure and plot it
import matplotlib.pyplot as plt
import pickle

# Create a figure
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])

# Save the figure
with open('my_figure.pkl', 'wb') as file:
    pickle.dump(fig, file)

del fig
del ax

