from data import generate_linear_data
from visual import plot_data

def main():
    x, y = generate_linear_data(n=100)
    plot_data(x, y, title='Linear Data')

if __name__ == "__main__":
    main()
