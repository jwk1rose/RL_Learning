import time
from typing import Optional, Union, List, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType
np.random.seed(1)
import render


def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridEnv(gym.Env):

    def __init__(self, size: int, target: Union[list, tuple, np.ndarray], forbidden: Union[list, tuple, np.ndarray],
                 render_mode: str):
        """
        GridEnv 的构造函数
        :param size: grid_world 的边长
        :param target: 目标点的pos
        :param forbidden: 不可通行区域 二维数组 或者嵌套列表 如 [[1,2],[2,2]]
        :param render_mode: 渲染模式 video表示保存视频
        """
        # 初始化可视化
        self.agent_location = np.array([0, 0])
        self.time_steps = 0
        self.size = size
        self.render_mode = render_mode
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)
        # 初始化起点 障碍物 目标点
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))
        self.target_location = np.array(target)
        # 初始化 动作空间 观测空间
        self.action_space, self.action_space_size = spaces.Discrete(5), spaces.Discrete(5).n
        self.reward_list = [0, 1, -10, -10]
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        # action to pos偏移量 的一个map
        self.action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }
        # Rsa表示 在 指定 state 选取指点 action 得到reward的概率
        self.Rsa = None
        # Psa表示 在 指定 state 选取指点 action 跳到下一个state的概率
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        reward = self.reward_list[self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)]
        direction = self.action_to_direction[action]
        self.render_.upgrade_agent(self.agent_location, direction, self.agent_location + direction)
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        terminated = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = self.get_info()
        return observation, reward, terminated, False, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        self.render_.show_frame(0.3)
        return None

    def get_obs(self) -> ObsType:
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def get_info(self) -> dict:
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        """
        初始化网格世界的 psa 和 rsa
        赵老师在b站评论区回答过 关于 rsa设计的问题
        原问题是；
        B友：老师您好，在spinning up 7.2.5里有写到
        Reward depends on the current state of the world, the action just taken, and the next state of the world.
        但您提到Rewad depends on the state and action, but not the next state.不知道reward 和 next state的关系是怎样的？

        答案如下：
        赵老师：这是一个很细小、但是很好的问题，说明你思考了。也许其他人也会有这样的疑问，我来详细解答一下。
        1）从贝尔曼公式和数学的角度来说，r是由p(r|s,a)决定的，所以从数学的角度r依赖于s,a，而不依赖于下一个状态s’。这是很简明的。
        2）举例，如果在target state刚好旁边是墙，agent试图撞墙又弹回来target state，这时候不应该给正r，而应该是给负r，因为r依赖于a而不是下一个状态。
        3）但是r是否和s’无关呢？实际是有关系的，否则为什么每次进到target state要得到正r呢？不过，这也可以等价理解成是在之前那个状态采取了好的动作才得到了正r。
        总结：r确实和s’有关，但是这种关系被设计蕴含到了条件概率p(r|s,a)中去。
        故而这里的rsa蕴含了next_state的信息
        :return:
        """
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(self.size ** 2, self.action_space_size, len(self.reward_list)), dtype=float)
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index)
                next_pos = pos + self.action_to_direction[action_index]
                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 3] = 1

                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if np.array_equal(next_pos, self.target_location):
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 0] = 1

    def close(self):
        pass


if __name__ == "__main__":
    grid = GridEnv(size=5, target=[1, 2], forbidden=[[2, 2]], render_mode='')
    grid.render()
