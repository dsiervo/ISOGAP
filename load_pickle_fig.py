# script to load a pickle matplot lib figure and plot it
import matplotlib.pyplot as plt
import pickle

# Load the figure from the file
with open('my_figure.pkl', 'rb') as file:
    fig2 = pickle.load(file)

# Instead of fig.show(), use plt.show() to display the figure
plt.show()
