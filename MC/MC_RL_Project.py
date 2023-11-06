import sys
import gym
import numpy as np
from collections import defaultdict

env = gym.make('Blackjack-v1')

print(env.observation_space)
print(env.action_space)

def generate_episode_from_Q(env, Q, epsilon, nA):

    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def get_probs(Q_s, epsilon, nA):

    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def update_Q(env, episode, Q, alpha, gamma):

    states, actions, rewards = zip(*episode)

    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n

    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start

    for i_episode in range(1, num_episodes+1):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        epsilon = max(epsilon*eps_decay, eps_min)

        episode = generate_episode_from_Q(env, Q, epsilon, nA)

        Q = update_Q(env, episode, Q, alpha, gamma)

    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q

policy, Q = mc_control(env, 500000, 0.02)

V = dict((k,np.max(v)) for k, v in Q.items())
