# Reinforce learning (RL)

# One of the three basic paradigm in ML along with supervise, unsupervise learning.
# It concerning how an  intelligent agent should take action in an 
# environment inorder to maximize the notion of cumulative reward. It doe not require 
# label and neither require the suu-optimal action to be explicitly corrected. Instead,
# it focus on finding a balance between exploration of uncharted territory and exploit of 
# current knowledge. 
# Partial RL is another way of combining the advantage of supervise learning and RL together.

# Many RL algorithm uses dynamic programming technique in the form of Markov decision process (MDP)
# The main different is that RL algorithm does not have knowledge of exat mathematical model 
# of MDP and usuall target toward large MDP where exact mehods is not feasable.

# RL is most closely to AI and feel like engineering ,science and social science. 
# while supervise and unsupervise learning feel like extension of data analsis.


# Markov (chain) decision Process
# A mathematicl framework forr modeling decision making process where outcome only partially 
# under control of decision maker or it is random.It's used to make sequential decision under 
# uncertainty random and partial control.It's the basis of RL today. mathematically, MDP is a 
# tuple of 5 elements (S,A,T,R,gamma) where 
# S -> set of state or state space 
# A -> action 
# T -> transition function it's defined the probability of transition to each state given state, action 
#         of previous time. 
# R -> reward function map every possible transition to real reward value 
# gamma -> discount factor defines relative weight of current and future reward

# MDP algorithm 
# There are many algorithms in MDP one such algorithm is value iteration algorithm VIL

# MDP goal is to choose sequential action that the agent can take to maximize the reward

# Value iteration algorthm uses iterative process to calculaate the value of each state using 
# Bellman equation 

# Bellman equation technique:
#         use the 
#         Bellman_eq = immediate reward + discounnted sum of future reward.
#         for each state to maximize the reward.
#         to generated the policy (π) to map state to action which guide the agent through
#         the uncertainty world toward its goal.

# In Julia the package POMDP is the interface package of reinforce learning to many other packages

# RL with MDP
# problem statement
# an agent was place on a 3x4 grid the probability of agent move to upper grid is 70% all other move
# is 10%. the reward is at the upper right corner but there penalty in lower grid next to it.
# there is a wall-tile to left
# try to get the best route toward the reward without getting penalize.

# loading the inerface
using POMDPs,POMDPModelTools,QuickPOMDPs
# load the solver in one of the package
using DiscreteValueIteration

# define grid state (tile)
struct State
    x::Int
    y::Int
end

# define action that agent can take (in Base.Enums)
@enum Action UP DOWN LEFT RIGHT

# define state space 
null = State(-1,-1)
S = [[State(x,y) for x = 1:4,y = 1:3]...,null]

# define action space
A  = [UP,DOWN,LEFT,RIGHT]

# define transition function that define how agent move from one state to another 
# given the action
# defind dictionary that map each action to new state 
const MOVEMENTS = Dict(
                    UP => State(0,1),
                    DOWN => State(0,-1),
                    LEFT => State(-1,0),
                    RIGHT => State(1,0)
                    )
# define a new method for + operator for use with state 
Base.:+(s1::State,s2::State) = State(s1.x + s2.x,s1.y + s2.y)

# define transitional function and use the deterministic() funcion 
# from the POMDPModelTools which use when distribution is required but the outcome
# is deterministic in this case is used to end the game
# this function specific to the problem
function T(s::State,a::Action)
    if R(s) != 0
        return Deterministic(null) # end game if agent land on +,- reward
    end
    # init variable use index 1 is current state
    len_a = length(A)
    next_states = Vector{State}(undef,len_a + 1) # 1 the current state,others possible new state
    probabilities = zeros(len_a + 1) # probab of move in any direction

    # enumerate() from Julia Base.Iterators
    # determine how the agent move inside the environment
    # walk down the list check the action movement
    for (index,a_prime) in enumerate(A)
        prob = (a_prime == a) ? 0.7 : 0.1 # probabilities agent move: right way 70%, not 10%
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest 
        
        if dest.x == 2 && dest.y == 2       # agent is in 2,2 tile grid
            probabilities[index + 1] = 0    # with the probabilities of movement
        elseif 1 <= dest.x <= 4 && 1 <= dest.y <= 3 # movement with in border
            probabilities[index + 1] += prob
        end
    end

    # handle the agent movement outside of border out-of-bound transition to remain in current state 
    next_states[1] = s 
    probabilities[1] = 1 - sum(probabilities)
    # sparsecategorical from POMDPModelTools
    # assign probabilities to next state
    return SparseCat(next_states,probabilities)
end

# define reward funcion 
function R(s, a = missing)
    if s == State(4,2)
        return  -100
    elseif s == State(4,3)
        return 10
    end
    return 0
end

# define discounnted factor higher for near term reward smaller for future reward
gamma = 0.95

#define MDP 
termination(s::State) = s == null 

abstract type GridWorld <: MDP{State,Action} end

# define MDP with 5 elements tuple
mdp = QuickMDP(GridWorld,
                states = S,
                actions = A,
                transition = T,
                reward = R,
                discount = gamma,
                isterminal = termination 
            )

# select the solver from DiscreteValueIteration value 30 seems to be best 

solver = ValueIterationSolver(max_iterations = 30)

# generate the polycy to solve the mdp 
policy =  solve(solver, mdp)

# view the values of the policy
value_view = [S policy.util]
