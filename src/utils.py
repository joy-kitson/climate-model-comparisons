import matplotlib.pyplot as plt

def save_or_show(save_as=None):
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)

def read_lines(path):
    with open(path, 'r') as in_file:
        return [l.strip() for l in in_file]

def write_lines(lines, path):
    with open(path, 'w') as out_file:
        out_file.writelines([l + '\n' for l in lines])
