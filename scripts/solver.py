import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

import grid_env
from model import *


# np.random.seed(1)


class Solve:
    def __init__(self, env: grid_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

    def random_greed_policy(self):
        """
        生成随机的greedy策略
        :return:
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy

    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        迭代求解贝尔曼公式 得到 state value tolerance 和 steps 满足其一即可
        :param policy: 需要求解的policy
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止计算 此时若是policy iteration 则算法变为 truncated iteration
        :return: 求解之后的收敛值
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  # bootstrapping
                state_value_k[state] = value
        return state_value_k

    def policy_improvement(self, state_value):
        """
        是普通 policy_improvement 的变种 相当于是值迭代算法 也可以 供策略迭代使用 做策略迭代时不需要 接收第二个返回值
        更新 qvalue ；qvalue[state,action]=reward+value[next_state]
        找到 state 处的 action*：action* = arg max(qvalue[state,action]) 即最优action即最大qvalue对应的action
        更新 policy ：将 action*的概率设为1 其他action的概率设为0 这是一个greedy policy
        :param: state_value: policy对应的state value
        :return: improved policy, 以及迭代下一步的state_value
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            state_value_k[state] = max(qvalue_list)
            action_star = qvalue_list.index(max(qvalue_list))
            policy[state, action_star] = 1
        return policy, state_value_k

    def calculate_qvalue(self, state, action, state_value):
        """
        计算qvalue elementwise形式
        :param state: 对应的state
        :param action: 对应的action
        :param state_value: 状态值
        :return: 计算出的结果
        """
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def value_iteration(self, tolerance=0.001, steps=100):
        """
        迭代求解最优贝尔曼公式 得到 最优state value tolerance 和 steps 满足其一即可
        :param tolerance: 当 前后 state_value 的范数小于tolerance 则认为state_value 已经收敛
        :param steps: 当迭代次数大于step时 停止 建议将此变量设置大一些
        :return: 剩余迭代次数
        """
        state_value_k = np.ones(self.state_space_size)
        while np.linalg.norm(state_value_k - self.state_value, ord=1) > tolerance and steps > 0:
            steps -= 1
            self.state_value = state_value_k.copy()
            self.policy, state_value_k = self.policy_improvement(state_value_k.copy())
        return steps

    def policy_iteration(self, tolerance=0.001, steps=100):
        """

        :param tolerance: 迭代前后policy的范数小于tolerance 则认为已经收敛
        :param steps: step 小的时候就退化成了  truncated iteration
        :return: 剩余迭代次数
        """
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            self.state_value = self.policy_evaluation(self.policy.copy(), tolerance, steps)
            self.policy, _ = self.policy_improvement(self.state_value)
        return steps

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def obtain_episode(self, policy, start_state, start_action, length):
        f"""

        :param policy: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :param length: episode 长度
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(policy[next_state])),
                                           p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode

    def mc_basic(self, length=30, epochs=10):
        """

        :param length: 每一个 state-action 对的长度
        :return:
        """
        for epoch in range(epochs):
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy, state, action, length)
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        g = episode[step]['reward'] + self.gama * g
                    self.qvalue[state][action] = g
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_star] = 1
            print(epoch)

    def mc_exploring_starts(self, length=10):
        time_start = time.time()
        policy = self.mean_policy.copy()
        qvalue = self.qvalue.copy()
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        while np.linalg.norm(policy - self.policy, ord=1) > 0.001:
            policy = self.policy.copy()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    visit_list = []
                    g = 0
                    episode = self.obtain_episode(policy=self.policy, start_state=state, start_action=action,
                                                  length=length)
                    for step in range(len(episode) - 1, -1, -1):
                        reward = episode[step]['reward']
                        state = episode[step]['state']
                        action = episode[step]['action']
                        g = self.gama * g + reward
                        # first visit
                        if [state, action] not in visit_list:
                            visit_list.append([state, action])
                            returns[state][action].append(g)
                            qvalue[state, action] = np.array(returns[state][action]).mean()
                            qvalue_star = qvalue[state].max()
                            action_star = qvalue[state].tolist().index(qvalue_star)
                            self.policy[state] = np.zeros(shape=self.action_space_size).copy()
                            self.policy[state, action_star] = 1
            print(np.linalg.norm(policy - self.policy, ord=1))

        time_end = time.time()
        print("mc_exploring_starts cost time:" + str(time_end - time_start))

    def mc_epsilon_greedy(self, length=1000, epsilon=1, tolerance=1):
        norm_list = []

        time_start = time.time()
        qvalue = np.random.random(size=(self.state_space_size, self.action_space_size))
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        while True:
            if epsilon >= 0.01:
                epsilon -= 0.01
                print(epsilon)
            length = 20 + epsilon * length
            if len(norm_list) >= 3:
                if norm_list[-1] < tolerance and norm_list[-2] < tolerance and norm_list[-3] < tolerance:
                    break

            # length = epsilon * length
            # if epsilon >= 0.01:
            #     epsilon -= 0.01
            qvalue = self.qvalue.copy()
            # for state in range(self.state_space_size):
            #     for action in range(self.action_space_size):
            state = random.choice(range(self.state_space_size))

            action = random.choice(range(self.action_space_size))

            episode = self.obtain_episode(policy=self.policy, start_state=state, start_action=action,
                                          length=length)
            g = 0
            for step in range(len(episode) - 1, -1, -1):
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                g = self.gama * g + reward
                # every visit
                returns[state][action].append(g)
                self.qvalue[state, action] = np.array(returns[state][action]).mean()
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon
            print(np.linalg.norm(self.qvalue - qvalue, ord=1))
            norm_list.append(np.linalg.norm(self.qvalue - qvalue, ord=1))

        time_end = time.time()
        print(len(norm_list))
        print("mc_exploring_starts cost time:" + str(time_end - time_start))

    def sarsa(self, alpha=0.1, epsilon=0.1, num_episodes=80):
        qvalue_list = [self.qvalue, self.qvalue + 1]
        while num_episodes > 0:
            done = False
            self.env.reset()
            next_state = 0
            num_episodes -= 1
            total_rewards = 0
            episode_length = 0
            # print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1), num_episodes)
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=self.policy[state])
                _, reward, done, _, _ = self.env.step(action)
                episode_length += 1
                total_rewards += reward
                next_state = self.env.pos2state(self.env.agent_location)
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p=self.policy[next_state])
                target = reward + self.gama * self.qvalue[next_state, next_action]
                error = self.qvalue[state, action] - target
                self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon
            # qvalue_list.append(self.qvalue.copy())

    def expected_sarsa(self, alpha=0.1, epsilon=1, num_episodes=1000):
        init_num = num_episodes
        qvalue_list = [self.qvalue, self.qvalue + 1]
        episode_index_list = []
        reward_list = []
        length_list = []
        while num_episodes > 0:
            if epsilon>0.1:
                epsilon-=0.01
            episode_index_list.append(init_num - num_episodes)
            done = False
            self.env.reset()
            next_state = 0
            total_rewards = 0
            episode_length = 0
            num_episodes -= 1
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1), num_episodes)
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=self.policy[state])
                _, reward, done, _, _ = self.env.step(action)
                next_state = self.env.pos2state(self.env.agent_location)
                expected_qvalue = 0
                episode_length += 1
                total_rewards += reward
                for next_action in range(self.action_space_size):
                    expected_qvalue += self.qvalue[next_state, next_action] * self.policy[next_state, next_action]
                target = reward + self.gama * expected_qvalue
                error = self.qvalue[state, action] - target
                self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon

            qvalue_list.append(self.qvalue.copy())
            reward_list.append(total_rewards)
            length_list.append(episode_length)
        fig = plt.figure(figsize=(10, 10))
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=reward_list, subplot_position=211,
                                            xlabel='episode_index', ylabel='total_reward')
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=length_list, subplot_position=212,
                                            xlabel='episode_index', ylabel='total_length')
        fig.show()

    def q_learning_on_policy(self, alpha=0.001, epsilon=0.4, num_episodes=1000):
        init_num = num_episodes
        qvalue_list = [self.qvalue, self.qvalue + 1]
        episode_index_list = []
        reward_list = []
        length_list = []
        while num_episodes > 0:
            episode_index_list.append(init_num - num_episodes)
            done = False
            self.env.reset()
            next_state = 0
            total_rewards = 0
            episode_length = 0
            num_episodes -= 1
            print(np.linalg.norm(qvalue_list[-1] - qvalue_list[-2], ord=1), num_episodes)
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=self.policy[state])
                _, reward, done, _, _ = self.env.step(action)
                next_state = self.env.pos2state(self.env.agent_location)
                episode_length += 1
                total_rewards += reward
                next_qvalue_star = self.qvalue[next_state].max()
                target = reward + self.gama * next_qvalue_star
                error = self.qvalue[state, action] - target
                self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon
            qvalue_list.append(self.qvalue.copy())
            reward_list.append(total_rewards)
            length_list.append(episode_length)
        fig = plt.figure(figsize=(10, 10))
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=reward_list, subplot_position=211,
                                            xlabel='episode_index', ylabel='total_reward')
        self.env.render_.add_subplot_to_fig(fig=fig, x=episode_index_list, y=length_list, subplot_position=212,
                                            xlabel='episode_index', ylabel='total_length')
        fig.show()

    def q_learning_off_policy(self, alpha=0.01, epsilon=0.1, num_episodes=2000, episode_length=2000):
        start_state = self.env.pos2state(self.env.agent_location)
        start_action = np.random.choice(np.arange(self.action_space_size),
                                        p=self.mean_policy[start_state])
        episode = self.obtain_episode(self.mean_policy.copy(), start_state=start_state, start_action=start_action,
                                      length=episode_length)
        for step in range(len(episode) - 1):
            reward = episode[step]['reward']
            state = episode[step]['state']
            action = episode[step]['action']
            next_state = episode[step + 1]['state']
            next_qvalue_star = self.qvalue[next_state].max()
            target = reward + self.gama * next_qvalue_star
            error = self.qvalue[state, action] - target
            self.qvalue[state, action] = self.qvalue[state, action] - alpha * error
            action_star = self.qvalue[state].argmax()
            self.policy[state] = np.zeros(self.action_space_size)
            self.policy[state][action_star] = 1

    def gfv(self, fourier: bool, state: int, ord: int) -> np.ndarray:
        """
        get_feature_vector
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """

        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")
        y, x = self.env.state2pos(state) + (1, 1)
        feature_vector = []
        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))

        else:
            # 归一化到 0 到 1
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(y_normalized ** (ord - i) * x_normalized ** j)

        return np.array(feature_vector)

    def gfv_a(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        """
        get_feature_vector_with_action
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """

        if state < 0 or state >= self.state_space_size or action < 0 or action >= self.action_space_size:
            raise ValueError("Invalid state/action value")
        feature_vector = []
        y, x = self.env.state2pos(state) + (1, 1)

        if fourier:
            # 归一化到 -1 到 1
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))

        else:
            # 归一化到 0 到 1
            state_normalized = (state - (self.state_space_size - 1) * 0.5) / (self.state_space_size - 1)
            action_normalized = (action - (self.action_space_size - 1) * 0.5) / (self.action_space_size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(state_normalized ** (ord - i) * action_normalized ** j)
        return np.array(feature_vector)

    def td_value_approximation(self, learning_rate=0.0005, epochs=100000, fourier=True, ord=5):
        self.state_value=self.policy_evaluation(self.policy)
        if not isinstance(learning_rate, float) or not isinstance(epochs, int) or not isinstance(
                fourier, bool) or not isinstance(ord, int):
            raise TypeError("Invalid input type")
        if learning_rate <= 0 or epochs <= 0 or ord <= 0:
            raise ValueError("Invalid input value")
        episode_length = epochs
        start_state = np.random.randint(self.state_space_size)
        start_action = np.random.choice(np.arange(self.action_space_size),
                                        p=self.mean_policy[start_state])
        episode = self.obtain_episode(self.mean_policy, start_state, start_action, length=episode_length)
        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()
        w = np.random.default_rng().normal(size=dim)
        rmse = []
        value_approximation = np.zeros(self.state_space_size)
        for epoch in range(epochs):
            reward = episode[epoch]['reward']
            state = episode[epoch]['state']
            next_state = episode[epoch]['next_state']
            target = reward + self.gama * np.dot(self.gfv(fourier, next_state, ord), w)
            error = target - np.dot(self.gfv(fourier, state, ord), w)
            gradient = self.gfv(fourier, state, ord)
            w = w + learning_rate * error * gradient
            for state in range(self.state_space_size):
                value_approximation[state] = np.dot(self.gfv(fourier, state, ord), w)
            rmse.append(np.sqrt(np.mean((value_approximation - self.state_value) ** 2)))
            print(epoch)
        X, Y = np.meshgrid(np.arange(1, 6), np.arange(1, 6))
        Z = self.state_value.reshape(5, 5)
        Z1 = value_approximation.reshape(5, 5)
        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('State Value')
        z_min = -5
        z_max = -2
        ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Value Approximation')
        ax1.set_zlim(z_min, z_max)
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        plt.show()
        return value_approximation

    def sarsa_function_approximation(self, learning_rate=0.0005, epsilon=0.1, num_episodes=100000, fourier=True, ord=5):
        #BUG

        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()
        w = np.random.default_rng().normal(size=dim)

        qvalue_approximation = np.zeros((self.state_space_size, self.action_space_size))
        reward_list = []
        length_list = []
        rmse = []
        policy_rmse = []
        policy = self.mean_policy.copy()
        next_state = 0
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=num_episodes)

        for episode in range(num_episodes):
            # epsilon = (epsilon - 1 / num_episodes) if epsilon > 0 else 0
            done = False
            self.env.reset()
            total_rewards = 0
            episode_length = 0
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=policy[state])
                _, reward, done, _, _ = self.env.step(action)
                episode_length += 1
                total_rewards += reward
                next_state = self.env.pos2state(self.env.agent_location)
                next_action = np.random.choice(np.arange(self.action_space_size),
                                               p=policy[next_state])
                target = reward + self.gama * np.dot(self.gfv_a(fourier, next_state, next_action, ord), w)
                error = target - np.dot(self.gfv_a(fourier, state, action, ord), w)
                gradient = self.gfv_a(fourier, state, action, ord)
                w = w + learning_rate * error * gradient
                # for state in range(self.state_space_size):
                #     for action in range(self.action_space_size):
                qvalue_approximation[state, action] = np.dot(self.gfv_a(fourier, state, action, ord), w)

                qvalue_star = qvalue_approximation[state].max()
                action_star = qvalue_approximation[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        policy[state, a] = 1 / self.action_space_size * epsilon
            rmse.append(np.sqrt(np.mean((qvalue_approximation - self.qvalue) ** 2)))
            # policy_rmse.append(np.sqrt(np.mean((policy - self.policy) ** 2)))
            reward_list.append(total_rewards)
            length_list.append(episode_length)
            print("episode={},length={},reward={}".format(episode, episode_length, total_rewards))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.plot(reward_list)
        ax.set_ylabel('total_reward')
        ax1 = fig.add_subplot(212)
        ax1.plot(length_list)
        ax1.set_xlabel('episode index')
        ax1.set_ylabel('episode length')
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)
        ax_rmse.plot(rmse, label='qvalue')
        # ax_rmse.plot(policy_rmse,label='policy')

        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        X, Y = np.meshgrid(np.arange(0, self.action_space_size), np.arange(0, self.state_space_size))
        Z = self.qvalue
        Z1 = qvalue_approximation
        print(Z.shape, Z1.shape, X.shape)

        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('q Value')
        # z_min = -5
        # z_max = -2
        # ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('qValue Approximation')
        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)

        plt.show()
        return qvalue_approximation

    def qlearning_function_approximation(self, learning_rate=0.0005, epsilon=0.1, num_episodes=100000, fourier=True,
                                         ord=15):
        #BUG

        dim = (ord + 1) ** 2 if fourier else np.arange(ord + 2).sum()
        w = np.random.default_rng().normal(size=dim)

        qvalue_approximation = np.zeros((self.state_space_size, self.action_space_size))
        reward_list = []
        length_list = []
        rmse = []
        policy = self.mean_policy.copy()
        next_state = 0
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=num_episodes)

        for episode in range(num_episodes):
            # epsilon = (epsilon - 1 / num_episodes) if epsilon > 0 else 0
            done = False
            self.env.reset()
            total_rewards = 0
            episode_length = 0
            while not done:
                state = next_state
                action = np.random.choice(np.arange(self.action_space_size),
                                          p=policy[state])
                _, reward, done, _, _ = self.env.step(action)
                episode_length += 1
                total_rewards += reward
                next_state = self.env.pos2state(self.env.agent_location)
                q_list = []
                for a in range(self.action_space_size):
                    q_list.append(np.dot(self.gfv_a(fourier, next_state, a, ord), w))

                target = reward + self.gama * np.array(q_list).max()
                error = target - np.dot(self.gfv_a(fourier, state, action, ord), w)
                gradient = self.gfv_a(fourier, state, action, ord)
                w = w + learning_rate * error * gradient
                for s in range(self.state_space_size):
                    for a in range(self.action_space_size):
                        qvalue_approximation[s, a] = np.dot(self.gfv_a(fourier, s, a, ord), w)
                qvalue_star = qvalue_approximation[state].max()
                action_star = qvalue_approximation[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        policy[state, a] = 1 / self.action_space_size * epsilon
            self.writer.add_scalar('rmse', np.sqrt(np.mean((qvalue_approximation - self.qvalue) ** 2)), episode)
            self.writer.add_scalar('episode_length', episode_length, episode)
            self.writer.add_scalar('total_reward', total_rewards, episode)

            # policy_rmse.append(np.sqrt(np.mean((policy - self.policy) ** 2)))
            # reward_list.append(total_rewards)
            # length_list.append(episode_length)
            print("episode={},length={},reward={}".format(episode, episode_length, total_rewards))

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(211)
        # ax.plot(reward_list)
        # ax.set_ylabel('total_reward')
        # ax1 = fig.add_subplot(212)
        # ax1.plot(length_list)
        # ax1.set_xlabel('episode index')
        # ax1.set_ylabel('episode length')
        # fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        # ax_rmse = fig_rmse.add_subplot(111)
        # ax_rmse.plot(rmse, label='qvalue')
        # # ax_rmse.plot(policy_rmse,label='policy')
        #
        # ax_rmse.set_title('RMSE')
        # ax_rmse.set_xlabel('Epoch')
        # ax_rmse.set_ylabel('RMSE')
        X, Y = np.meshgrid(np.arange(0, self.action_space_size), np.arange(0, self.state_space_size))
        Z = self.qvalue
        Z1 = qvalue_approximation
        print(Z.shape, Z1.shape, X.shape)

        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('q Value')
        z_min = -6
        z_max = 0
        ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('qValue Approximation')
        ax1.set_zlim(z_min, z_max)

        # fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        # ax_rmse = fig_rmse.add_subplot(111)
        self.writer.close()
        plt.show()
        return qvalue_approximation

    def qvalue_function_approximation(self, learning_rate=0.00008, epsilon=0.1, num_episodes=1000000,
                                      fourier=True,
                                      ord=5):
        #BUG
        dim = (ord + 1) ** 3 if fourier else np.arange(ord + 2).sum()
        w = np.random.default_rng().normal(size=dim)
        qvalue_approximation = np.zeros(shape=(self.state_space_size, self.action_space_size))
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=100000)

        for epoch in range(num_episodes):
            # epsilon = (epsilon - 1 / num_episodes) if epsilon > 0 else 0
            step = int(np.random.randint(low=0, high=99999, size=1))
            reward = episode[step]['reward']
            state = episode[step]['state']
            action = episode[step]['action']
            next_action = episode[step]['next_action']
            next_state = episode[step]['next_state']
            target = reward + self.gama * np.dot(self.gfv_a(fourier, next_state, next_action, ord), w)
            error = target - np.dot(self.gfv_a(fourier, state, action, ord), w)
            gradient = self.gfv_a(fourier, state, action, ord)
            w = w + learning_rate * error * gradient
            for a in range(self.action_space_size):
                qvalue_approximation[state, a] = np.dot(self.gfv_a(fourier, state, a, ord), w)
            self.writer.add_scalar('rmse', np.sqrt(np.mean((qvalue_approximation - self.qvalue) ** 2)), epoch)
            if epoch % 1000 == 0:
                print(epoch, np.sqrt(np.mean((qvalue_approximation - self.qvalue) ** 2)))
        X, Y = np.meshgrid(np.arange(0, self.action_space_size), np.arange(0, self.state_space_size))
        Z = self.qvalue
        Z1 = qvalue_approximation
        print(Z.shape, Z1.shape, X.shape)

        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('q Value')
        z_min = -6
        z_max = 0
        ax.set_zlim(z_min, z_max)
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('qValue Approximation')
        ax1.set_zlim(z_min, z_max)
        for i in range(self.state_space_size):
            for j in range(self.action_space_size):
                print("qvalue:{},approximation:{}".format(self.qvalue[i, j], qvalue_approximation[i, j]))

        self.writer.close()
        plt.show()
        return qvalue_approximation

    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """构造一个PyTorch数据迭代器"""
        reward = []
        state_action = []
        next_state = []
        for i in range(len(episode)):
            reward.append(episode[i]['reward'])
            action = episode[i]['action']
            y, x = self.env.state2pos(episode[i]['state'])
            state_action.append((y, x, action))
            y, x = self.env.state2pos(episode[i]['next_state'])
            next_state.append((y, x))
        reward = torch.tensor(reward).reshape(-1, 1)
        state_action = torch.tensor(state_action)
        next_state = torch.tensor(next_state)
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)

    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=600, batch_size=100, update_step=10):
        q_net = QNET()
        policy = self.policy.copy()
        state_value = self.state_value.copy()
        q_target_net = QNET()
        q_target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.SGD(q_net.parameters(),
                                    lr=learning_rate)
        episode = self.obtain_episode(self.mean_policy, 0, 0, length=episode_length)
        date_iter = self.get_data_iter(episode, batch_size)
        loss = torch.nn.MSELoss()
        approximation_q_value = np.zeros(shape=(self.state_space_size, self.action_space_size))
        i = 0
        rmse_list=[]
        loss_list=[]
        for epoch in range(epochs):
            for state_action, reward, next_state in date_iter:
                i += 1
                q_value = q_net(state_action)
                q_value_target = torch.empty((batch_size, 0))  # 定义空的张量
                for action in range(self.action_space_size):
                    s_a = torch.cat((next_state, torch.full((batch_size, 1), action)), dim=1)
                    q_value_target = torch.cat((q_value_target, q_target_net(s_a)), dim=1)
                q_star = torch.max(q_value_target, dim=1, keepdim=True)[0]
                y_target_value = reward + self.gama * q_star
                l = loss(q_value, y_target_value)
                optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0S
                l.backward()  # 反向传播更新参数
                optimizer.step()
                if i % update_step == 0 and i != 0:
                    q_target_net.load_state_dict(
                        q_net.state_dict())  # 更新目标网络
                    # policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            loss_list.append(float(l))
            print("loss:{},epoch:{}".format(l, epoch))
            self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            self.state_value = np.zeros(shape=self.state_space_size)

            for s in range(self.state_space_size):
                y, x = self.env.state2pos(s)
                for a in range(self.action_space_size):
                    approximation_q_value[s, a] = float(q_net(torch.tensor((y, x, a)).reshape(-1, 3)))
                q_star_index = approximation_q_value[s].argmax()
                self.policy[s, q_star_index] = 1
                self.state_value[s] = approximation_q_value[s, q_star_index]
            rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))
            # policy_rmse = np.sqrt(np.mean((policy - self.policy) ** 2))
        fig_rmse = plt.figure(figsize=(8, 12))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(211)

        # 绘制 rmse 图像
        ax_rmse.plot(rmse_list)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        self.writer.close()
        ax_loss = fig_rmse.add_subplot(212)

        ax_loss.plot(loss_list)
        ax_loss.set_title('loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        plt.show()

    def obtain_episode_p(self, policy_net, start_state, start_action):
        f"""

        :param policy_net: 由指定策略产生episode
        :param start_state: 起始state
        :param start_action: 起始action
        :return: 一个 state,action,reward,next_state,next_action 序列
        """
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        done = False
        while not done:
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            y, x = self.env.state2pos(next_state) / self.env.size
            prb = policy_net(torch.tensor((y, x)).reshape(-1, 2))[0]

            next_action = np.random.choice(np.arange(self.action_space_size),
                                           p=prb.detach().numpy())
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode

    def reinforce(self, learning_rate=0.001, epochs=20000, episode_length=100):
        policy_net = PolicyNet()
        optimizer = torch.optim.Adam(policy_net.parameters(),
                                     lr=learning_rate)
        for epoch in range(epochs):
            # start_state =  0
            # y, x = self.env.state2pos(start_state) / self.env.size
            prb = policy_net(torch.tensor((0, 0)).reshape(-1, 2))[0]
            start_action = np.random.choice(np.arange(self.action_space_size),
                                            p=prb.detach().numpy())
            episode = self.obtain_episode_p(policy_net, 0, start_action)
            if (len(episode) < 10):
                g = -100
            else:
                g = 0
            optimizer.zero_grad()

            for step in reversed(range(len(episode))):

                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                if len(episode) > 1000:
                    print(g, reward)
                g = self.gama * g + reward
                self.qvalue[state, action] = g
                y, x = self.env.state2pos(state) / self.env.size
                prb = policy_net(torch.tensor((y, x)).reshape(-1, 2))[0]
                log_prob = torch.log(prb[action])
                loss = -log_prob * g
                loss.backward()  # 反向传播计算梯度
            self.writer.add_scalar('loss', float(loss.detach()), epoch)
            self.writer.add_scalar('g', g, epoch)
            self.writer.add_scalar('episode_length', len(episode), epoch)
            print(epoch, len(episode), g)
            optimizer.step()
        for s in range(self.state_space_size):
            y, x = self.env.state2pos(s) / self.env.size
            prb = policy_net(torch.tensor((y, x)).reshape(-1, 2))[0].detach().numpy()
            self.policy[s, :] = prb.copy()
        self.writer.close()


if __name__ == "__main__":
    # env = grid_env.GridEnv(size=2, target=[1, 1], forbidden=[[1, 0]],
    #                        render_mode='')

    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    # env = grid_env.GridEnv(size=3, target=[2, 1], forbidden=[[2, 0], [1, 0], [1, 1]],
    #                        render_mode='')
    solver = Solve(env)
    # solver.show_state_value(solver.state_value, y_offset=0.2)
    solver.q_learning_off_policy()
    solver.state_value = solver.policy_evaluation(solver.policy, steps=100)

    start_time = time.time()
    solver.dqn()
    # solver.q_learning_off_policy()

    # solver.td_value_approximation()

    # solver.show_state_value(state_value=solver.td_value_approximation(), y_offset=-0.25)
    # solver.q_learning_on_policy()
    # solver.qvalue_function_approximation()
    # solver.qlearning_function_approximation()
    # solver.dqn()
    # solver.reinforce()
    end_time = time.time()

    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(env.render_.trajectory))
    # solver.mc_epsilon_greedy()
    # solver.mc_exploring_starts(length=20)
    # solver.mc_basic(length=15)
    solver.show_policy()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)

    # solver.env.render()
    # solver.env.render_.draw_episode()
    solver.env.render()
