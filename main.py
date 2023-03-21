import numpy as np
import pandas as pd
from scipy import stats
from time import perf_counter as p_count
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


def perform_task1(arr_len: int = 10 ** 6, experiments_cnt: int = 100) -> None:
    """
    First task was implemented with a little upgrade: it uses some parameters to
    perform an experiment in better way.

    :param arr_len: length of arrays -> int
    :param experiments_cnt: number of experiments -> int
    :return: None
    """
    print('TASK №1', end='\n\n')
    times = np.array([[], []], float)

    print(f'Experiment with list and numpy.arrays multiplication'
          f'\n\titeration number: {experiments_cnt}'
          f'\n\tsize of generated objects: {arr_len}')

    for i in range(experiments_cnt):
        arr_lst_1 = [random() for __ in range(arr_len)]
        arr_lst_2 = [random() for __ in range(arr_len)]
        out_lst = list()
        arr_np_1 = np.array(arr_lst_1, float)
        arr_np_2 = np.array(arr_lst_2, float)
        out_np = np.array([], float)
        t = p_count()
        for el1, el2 in zip(arr_lst_1, arr_lst_2):
            out_lst.append(el1 * el2)
        t1 = p_count() - t
        del out_lst

        t = p_count()
        out_np = np.multiply(arr_np_1, arr_np_2)
        t2 = p_count() - t
        del out_np
        times = np.append(times, [[t1], [t2]], axis=1)
        print(f'# processing... {i / experiments_cnt * 100:>.0f}%', end='\r')
    print('# done! 100%')

    print(f'RESULTS:'
          f'\n\tlist results: {times[0].min():>.4f}-{times[0].max():>.4f} sec.'
          f'\n\tnumpy results: {times[1].min():>.4f}-{times[1].max():>.4f} sec.'
          f'\n!!\\ numpy is {(times[0] / times[1]).mean():>.1f}x times faster as average result')


def perform_task2(csv_fpath: str = 'data2.csv', col_i: int = 5) -> None:
    """
    Draws two bars by data from selected column index. First one is
    raw data view, another one is normal form representation.
    If no data passed, this function uses default values for the variant №10.

    :param csv_fpath: file name or full path -> str
    :param col_i: column index -> int
    """
    def _set_words(title: str, x: str, y: str):
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

    data = pd.read_csv(csv_fpath, sep=',')
    colname = data.columns[col_i]
    fig = plt.figure(figsize=(10, 4))
    plt.suptitle(colname)
    # raw subplot
    sub1 = fig.add_subplot(121)
    _set_words('Raw data', 'element\'s index', 'S/m')
    data = data[colname]
    sub1.bar(data.index, data)
    sub1.grid()
    # normal
    sub2 = fig.add_subplot(122)
    _set_words(f'Normalized data, std={data.std():.2f}', 'S/m', 'frequency')
    data.hist(bins=int(data.max() / data.std()), grid=True)
    plt.show()


def perform_task3():
    """
    Draws f(x)={y=x*cos*(x); z=sin(x)} at [-3pi;3pi]
    """
    np.random.seed(40)
    xs = np.linspace(-3*np.pi, 3*np.pi, 1000)
    ys = xs * np.cos(xs)
    zs = np.sin(xs)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, c='red')
    ax.set_title('f(x) = {y=x*cos*(x); z=sin(x)}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def perform_task_additional():
    def animate_func(i):
        ax.clear()
        ax.plot(coord[0, :i + 1], coord[1, :i + 1], c='red')
        ax.plot(coord[0, i], coord[1, i], marker='o', c='red')
        ax.set_title('y = sin(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 20])
        ax.set_ylim([-2, 2])

    x = np.linspace(0, 20, 100)
    y = np.sin(x)
    coord = np.array([x, y])

    fig = plt.figure()
    ax = plt.axes()
    line = animation.FuncAnimation(
        fig, animate_func, interval=100, frames=len(x)
    )
    plt.show()


if __name__ == '__main__':
    if 'y' == input('Start Task №1? (y/n)\n>> ').lower():
        perform_task1(10 ** 6, 20)
    if 'y' == input('Start Task №2? (y/n)\n>> ').lower():
        perform_task2()
    if 'y' == input('Start Task №3? (y/n)\n>> ').lower():
        perform_task3()
    if 'y' == input('Start Task Additional? (y/n)\n>> ').lower():
        perform_task_additional()
