from arguments import get_args
from algo import *
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc

t = time.time()


def plot(record, info, convergence_time, para=''):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    import os
    dir = str(t) + '-{}'.format(info) + para
    os.makedirs(dir, exist_ok=True)
    fig.savefig(dir + '/performance.png')
    with open(dir + '/convergence.txt', 'a') as f:
        if convergence_time < 5 and record['min'][-1] > 85:
            f.write('{} {}\n'.format(record['steps'][-1], time.strftime('%Mm %Ss', time.gmtime(time.time() - t))))
            convergence_time += 1
    plt.close()
    return convergence_time


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)

    # agent initial
    # you should finish your agent with QAgent
    # e.g. agent = myQAgent()
    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99
    n = args.n
    m = args.m
    h = args.h
    start_planning = args.sp
    agent = MyQAgent(lr=alpha, discount=gamma)
    model_type = args.model
    is_nn = True
    is_clip = False
    if model_type == 'nn':
        dynamics_model = NetworkModel(8, 8, policy=agent)
    elif model_type == 'advnn':
        dynamics_model = AdvNetworkModel(8, 8, policy=agent)
        is_clip = True
    elif model_type == 'dyna':
        dynamics_model = DynaModel(8, 8, policy=agent)
        is_nn = False
    else:
        print('Unable to identify model type: {}'.format(str(model_type)))

    params = {
        '-n=': n,
        '-m=': m,
        '-h=': h,
        '-sp=': start_planning
    }
    suffix = ''
    for k, v in params.items():
        suffix += k + str(v)
    convergence_time = 0

    # start to train your agent
    for i in range(num_updates):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        for step in range(args.num_steps):
            # Sample actions with epsilon greedy policy

            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            obs_next = obs_next.astype(int)
            agent.update_table(obs, action, obs_next, reward, done, is_clip)
            # add your Q-learning algorithm
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next

            if done:
                obs = envs.reset()
        if i > start_planning:
            for _ in range(n):
                s, idx = dynamics_model.sample_state()
                # buf_tuple = dynamics_model.buffer[idx]
                for _ in range(h):
                    if is_nn:
                        if np.random.rand() < epsilon:
                            a = envs.action_sample()
                        else:
                            a = agent.select_action(s)
                    else:
                        a = dynamics_model.sample_action(s)
                    s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    agent.update_table(s, a, s_, r, done, is_clip)
                    s = s_
                    if done:
                        break

        for _ in range(m):
            dynamics_model.train_transition(32)

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                i, total_num_steps, int(total_num_steps / (end - start)),
                np.mean(reward_episode_set),
                np.min(reward_episode_set),
                np.max(reward_episode_set)))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            convergence_time = plot(record, args.info, convergence_time, suffix)


if __name__ == "__main__":
    main()
