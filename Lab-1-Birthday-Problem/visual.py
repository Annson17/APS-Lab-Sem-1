import matplotlib.pyplot as plt

def plot_data(x, y,title='Linear Data'):
    plt.scatter(x[:,0], x[:,1], c=y, cmap='bwr', alpha=0.7)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()