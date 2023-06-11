import random
import time

import matplotlib.pyplot as plt
import numpy as np

import grid_env


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

    def show_state_value(self):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(self.state_value[state], 1)),
                                        y_offset=0.2,
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

    def mc_basic(self, length=5):
        """

        :param length: 每一个 state-action 对的长度
        :return:
        """
        time_start = time.time()
        policy = self.mean_policy.copy()
        while np.linalg.norm(policy - self.policy, ord=1) > 0.001:
            self.policy = policy.copy()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(policy, state, action, length)
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        g = episode[step]['reward'] + self.gama * g
                    self.qvalue[state][action] = g
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                policy[state] = np.zeros(shape=self.action_space_size)
                policy[state, action_star] = 1
            print(np.linalg.norm(policy - self.policy, ord=1))
        time_end = time.time()
        print("mc_basic cost time:" + str(time_end - time_start))

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

    def mc_epsilon_greedy(self, length=100, epsilon=1, tolerance=1):
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

    def sarsa(self, alpha=0.1, epsilon=0.1, num_episodes=500):
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

    def expected_sarsa(self, alpha=0.1, epsilon=0.1, num_episodes=1000):
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

    def q_learning_on_policy(self, alpha=0.1, epsilon=0.7, num_episodes=1000):
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

    def q_learning_off_policy(self, alpha=0.01, epsilon=0.1, num_episodes=2000, episode_length=5000):
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


if __name__ == "__main__":
    env = grid_env.GridEnv(size=5, target=[2, 3], forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    solver = Solve(env)
    solver.q_learning_off_policy()
    # solver.mc_epsilon_greedy()
    # solver.mc_exploring_starts(length=20)
    # solver.mc_basic(length=15)
    solver.state_value = solver.policy_evaluation(solver.policy, steps=100)
    solver.show_policy()
    solver.show_state_value()
    solver.env.render()
    solver.env.render_.draw_episode()
    solver.env.render()
