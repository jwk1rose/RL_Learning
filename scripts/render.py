from typing import Union

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
class Render:
    def __init__(self, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 size=5):
        """ Render 类的构造函数

        :param target:目标点的位置
        :param forbidden:障碍物区域位置
        :param size:网格世界的size 默认为 5x5
        """
        # 初始化
        self.agent = None
        self.target = target
        self.forbidden = forbidden
        self.size = size
        self.fig = plt.figure(figsize=(10, 10), dpi=self.size * 20)
        self.ax = plt.gca()
        self.ax.xaxis.set_ticks_position('top')
        self.ax.invert_yaxis()
        self.ax.xaxis.set_ticks(range(0, size + 1))
        self.ax.yaxis.set_ticks(range(0, size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labeltop=False)
        # 绘制网格世界的state index 以及grid边框的标号
        # index = 0
        for y in range(size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)
            # for x in range(size):
            #     self.write_word(pos=(x, y), word="s" + str(index), size_discount=0.65)
            #     index += 1
        # 填充障碍物和目标格子
        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos=self.target, color='darkturquoise')
        self.trajectory = []
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
        self.ax.add_patch(self.agent)

    def fill_block(self, pos: Union[list, tuple, np.ndarray], color: str = '#EDB120', width=1.0,
                   height=1.0) -> patches.RegularPolygon:
        """
        对指定pos的网格填充颜色
        :param width:
        :param height:
        :param pos: 需要填充的网格的左下坐标
        :param color: 填充的颜色 默认为‘EDB120’表示 forbidden 格子 ,‘#4DBEEE’表示 target 格子
        :return: Rectangle对象
        """
        return self.ax.add_patch(
            patches.Rectangle((pos[0], pos[1]),
                              width=1.0,
                              height=1.0,
                              facecolor=color,
                              fill=True,
                              alpha=0.90,
                              ))

    def draw_random_line(self, pos1: Union[list, tuple, np.ndarray], pos2: Union[list, tuple, np.ndarray]) -> None:
        """
        在pos1 和pos2之间生成一条线条，这条线条会在pos1和pos2之间产生随机偏移
        :param pos1: 起点所在位置的坐标
        :param pos2: 终点所在位置的坐标
        :return:None
        """
        offset1 = np.random.uniform(low=-0.05, high=0.05, size=1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        x = [pos1[0] + 0.5, pos2[0] + 0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        if pos1[0] == pos2[0]:
            x = [x[0] + offset1, x[1] + offset2]
        else:
            y = [y[0] + offset1, y[1] + offset2]
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)

    def draw_circle(self, pos: Union[list, tuple, np.ndarray], radius: float,
                    color: str = 'green', fill: bool = True) -> patches.CirclePolygon:
        """
        对指定pos的网格内画一个圆
        :param fill: 是否填充圆的内部
        :param radius: 圆的半径
        :param pos: 需要画圆的网格的左下坐标
        :param color: 'lime'表示 绿色
        :return: CirclePolygon
        """
        return self.ax.add_patch(
            patches.Circle((pos[0] + 0.5, pos[1] + 0.5),
                           radius=radius,
                           facecolor=color,
                           edgecolor='green',
                           linewidth=2,
                           fill=fill
                           ))

    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        """
        将动作可视化
        :param radius: circle 的半径
        :param pos:网格的左下坐标
        :param toward:(a,b) a b 分别表示 箭头在x方向和y方向的分量 如果是一个 0 向量就画圆
        :param color: 箭头的颜色 默认为green
        :return:None
        """
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            self.ax.add_patch(
                patches.Arrow(pos[0] + 0.5, pos[1] + 0.5, dx=toward[0],
                              dy=toward[1], color=color, width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                              linewidth=0.5))
        else:
            self.draw_circle(pos=tuple(pos), color='white', radius=radius, fill=False)

    def write_word(self, pos: Union[list, np.ndarray, tuple], word: str, color: str = 'black', y_offset: float = 0,
                   size_discount: float = 1.0) -> None:
        """
        在网格上对应位置写字
        :param pos: 需要写字的格子的左下角坐标
        :param word: 要写的字
        :param color: 字的颜色
        :param y_offset: 字在y方向上关于网格中心的偏移
        :param size_discount: 字体大小 (0-1)
        :return: None
        """
        self.ax.text(pos[0] + 0.5, pos[1] + 0.5 + y_offset, word, size=size_discount * (30 - 2 * self.size), ha='center',
                 va='center', color=color)

    def upgrade_agent(self, pos: Union[list, np.ndarray, tuple], action,
                      next_pos: Union[list, np.ndarray, tuple], ) -> None:
        """
        更新agent的位置
        :param next_pos: 当前pos和下一步的位置
        :param action: 对应位置采取的action
        :param pos: 当前的state位置
        :return: None
        """

        self.trajectory.append([tuple(pos), action, tuple(next_pos)])

    def show_frame(self, t: float = 0.2) -> None:
        """
        显示figure 持续一段时间后 关闭
        :param t: 持续时间
        :return: None
        """
        self.fig.show()

    def save_frame(self, name: str) -> None:
        """
        将当前帧保存
        :param name:保存的文件名
        :return: None
        """
        self.fig.savefig(name + ".jpg")

    def save_video(self, name: str) -> None:
        """
        如果指定了起点 想要将agent从起点到终点的轨迹show出来的话，可以使用这个函数保存视频
        :param name:视频文件的名字
        :return:None
        """
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init(), frames=len(self.trajectory),
                                       interval=25, repeat=False)
        anim.save(name + '.mp4')

    # init 和 animate 都是服务于animation.FuncAnimation
    # 具体用法参考matplotlib官网
    def init(self):
        pass

    def animate(self, i):
        print(i,len(self.trajectory))
        location = self.trajectory[i][0]
        action = self.trajectory[i][1]
        next_location = self.trajectory[i][2]
        next_location = np.clip(next_location, -0.4, self.size - 0.6)
        self.agent.remove()
        if action[0] + action[1] != 0:
            self.agent = patches.Arrow(x=location[0] + 0.5, y=location[1] + 0.5,
                                       dx=action[0] / 2, dy=action[1] / 2,
                                       color='b',
                                       width=0.5)
        else:
            self.agent = patches.Circle(xy=(location[0] + 0.5, location[1] + 0.5),
                                        radius=0.15, fill=True, color='b',
                                        )
        self.ax.add_patch(self.agent)

        self.draw_random_line(pos1=location, pos2=next_location)

    def draw_episode(self):
        for i in range(len(self.trajectory)):
            location = self.trajectory[i][0]
            next_location = self.trajectory[i][2]
            self.draw_random_line(pos1=location, pos2=next_location)

    def add_subplot_to_fig(self, fig, x, y, subplot_position, xlabel, ylabel, title=''):
        """
        在给定的位置上添加一个子图到当前的图中，并在子图中调用plot函数，设置x,y label和title。

        参数:
        x: 用于plot的x数据
        y: 用于plot的y数据
        subplot_position: 子图的位置，格式为 (row, column, index)
        xlabel: x轴的标签
        ylabel: y轴的标签
        title: 子图的标题
        """
        # 在指定位置添加子图
        ax = fig.add_subplot(subplot_position)
        # 调用plot函数绘制图形
        ax.plot(x, y)
        # 设置x,y label和title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)


if __name__ == '__main__':
    render = Render(target=[4, 4], forbidden=[np.array([1, 2]), np.array([2, 2])], size=5)
    render.draw_action(pos=[3, 3], toward=(0, 0.4))
    # render.save_frame('test1')

    for num in range(10):
        render.draw_random_line(pos1=[1.5, 1.5], pos2=[1.5, 2.5])

    action_to_direction = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
        4: np.array([0, 0]),
    }
    uniform_policy = np.random.random(size=(25, 5))
    # uniform_policy = np.ones(shape=(25, 5)) / 5
    # for state in range(25):
    #     for action in range(5):
    #         policy = uniform_policy[state, action]
    #         render.draw_action(pos=[state // 5, state % 5], toward=policy * 0.4 * action_to_direction[action],
    #                            radius=0.03 + 0.07 * policy)
    for a in range(5):
        render.trajectory.append((a, a))
    render.show_frame()
