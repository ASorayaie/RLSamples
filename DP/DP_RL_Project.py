import gym
import numpy as np
import matplotlib.pyplot as plt

name = 'FrozenLake-v1'
env = gym.make(name, is_slippery=False)
env.seed(0)
env.action_space.seed(0)

env.reset()

print('Action space: ' + str(env.action_space))
print('Reward range: ' + str(env.reward_range))
print('Observation space: ' + str(env.observation_space))

env.render()

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

env.reset()
next_state, reward, terminated, info = env.step(DOWN)
print('=============')
print('Next state: ' + str(next_state))
print('Terminated: ' + str(terminated))
print('Reward: ' + str(reward))
print('Info: ' + str(info))

def policy_evaluation(env, policy, gamma=1, theta=1e-8, draw=False):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s] - Vs))
            V[s] = Vs
        if draw: plot(V, policy, draw_vals=True)
        if delta < theta:
            break
    return V

policy = np.ones([env.nS, env.nA]) / env.nA

V = policy_evaluation(env, policy, draw=False)

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)

        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0) / len(best_a)

    return policy

policy = policy_improvement(env, V)
