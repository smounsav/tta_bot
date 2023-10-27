import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as T

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

def create_barplot(accs :dict, title :str, savepath :str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks = list(accs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f'{j:.1f}', ha='center', va='bottom', fontsize=7)

    plt.title(title)
    plt.ylabel('Accuracy (%)')

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis='y')
    plt.savefig(savepath)
    plt.close()


def get_fname(weight_path :str):
    return '.'.join(weight_path.split('/')[-1].split('.')[:-1])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def invNorm(images, mean, std):
    images = T.functional.normalize(images, 0, [1/i for i in std])
    images = T.functional.normalize(images, [-i for i in mean], 1)
    return images