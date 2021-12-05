from MDP import *

''' Construct simple MDP '''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
print(" ++++++ Value Iteration ++++++ ")
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print('optimal values from value iteration: ', V)
print('number of iterations: ', nIterations)
print('epsilon: ', epsilon)
print(" ++++++ Extract Policy ++++++ ")
policy = mdp.extractPolicy(V)
print('extracted policy: ', policy)
print(" ++++++ Evaluate Policy ++++++ ")
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print('policy evaluation for [1, 0, 1, 0]:', V)
V = mdp.evaluatePolicy(np.array([0, 1, 1, 1]))
print('policy evaluation for [0, 1, 1, 1]:', V)
print(" ++++++ Policy Iteration ++++++ ")
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print('policy: ', policy)
print('value functions: ', V)
print('number of iterations: ', iterId)