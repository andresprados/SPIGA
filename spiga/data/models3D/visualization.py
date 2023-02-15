import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Input arguments control
    pars = argparse.ArgumentParser(description='3D model visualization')
    pars.add_argument('file', type=str, help='File txt path')
    args = pars.parse_args()
    visualize_3Dmodel(args.file)


def visualize_3Dmodel(input_file):

    with open(input_file) as f:
        lines = f.readlines()

    model = []
    for line in lines:
        line = line[:-1]    # Remove \n
        line_split = line.split('|')
        values = np.array(line_split, dtype=float)
        model.append(values)

    model = np.array(model)
    model_xyz = model[:, 1:]

    # Show model
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_xyz[:, 0], model_xyz[:, 1], model_xyz[:, 2]+0.8)
    plt.show()


if __name__ == '__main__':
    main()
