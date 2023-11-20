import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


plt.rcParams.update({'font.size': 18})

#initialize the non-slippery Frozen Lake environment
environment = gym.make("FrozenLake-v1", is_slippery=False)
environment.reset()
environment.render()

qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

episodes = 1000

#learning rate
alpha = 0.5

#discount factor
gamma = 0.9

#Randomness factor
epsilon = 1.0

#epsilon decay
epsilon_decay = 0.001

outcomes = []

print('Q-table before training: ')
print(qtable)

#training
for _ in range(episodes):
    state = environment.reset()
    done = False

    outcomes.append('Failure')

    while not done:
        #Rand number between 0 and 1
        rnd = np.random.random()
        # print(f"Rand: {rnd}")
        # print(f"Epsilon: {epsilon}")

        #Rand < epsilon, take random action
        if rnd < epsilon:
            action = environment.action_space.sample()
        
        else:
            action = np.argmax(qtable[state])

        #implement action and the agent in desired direction
        new_state, reward, done, info = environment.step(action)

        #Update Q(s, a)
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma \
        * np.max(qtable[new_state]) - qtable[state, action])

        #Update current state
        state = new_state

        #Reward means success
        if reward:
            outcomes[-1] = "Success"
    epsilon = max(epsilon - epsilon_decay, 0)

    
        

print()
print('================')
print('Q-table after training:')
print(qtable)


plt.figure(figsize=(12,5))
plt.xlabel("Run number", fontsize=20)
plt.ylabel("Outcome", fontsize=20)
ax = plt.gca()
plt.bar(range(len(outcomes)), outcomes, width=1.0)
plt.show()

episodes = 100
nb_success = 0

#Evaluation
for _ in range(100):
    state = environment.reset()
    done = False

    #Keep training until agent gets stuck or reaches goal
    while not done:

        #choose action with highest value in current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        
        #If there's no best action (only zeros), take a random one
        else:
            action = environment.action_space.sample()

        new_state, reward, done, info = environment.step(action)

        state = new_state

        #When we get a reward, it means we solved the game
        nb_success += reward

print(f"Success rate = {nb_success/episodes*100}")
# print(outcomes)

# state = environment.reset()
# done = False
# sequence = []

# while not done:
#     if np.max(qtable[state]) > 0:
#         action = np.argmax(qtable[state])

#     else:
#         action = environment.action_space.sample()
    
#     #Add action to sequence
#     sequence.append(action)

#     new_state, reward, done, info = environment.step(action)

#     #Update our current state
#     state = new_state

#     #Update the render
#     clear_output(wait=True)
#     environment.render()
#     time.sleep(1)

# print(f"Sequence = {sequence}")
        


