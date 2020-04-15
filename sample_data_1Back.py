import numpy as np 
import pandas as pd
import scipy as sp
from pylab import * 

def getActionProb(q1, q2,  beta):
    return exp(beta*q1)/( exp(beta*q1) + exp(beta*q2))

## Generate PRL data
def generate_data_PRL(params,trial_length):
    [alpha,beta_param] = params
    [q_left, q_right] = [.5, .5]
    action_history=[]
    reward_history=[]
    win_shift=[]
    win_stay=[]
    lose_shift=[]
    lose_stay=[]
    num_correct_presses=0
    high_lever=1
    for i in range(trial_length):
        p_left = getActionProb(q_left, q_right, beta_param)
        p_right = 1 - p_left
        action = int(rand() < p_left)
        action_history.append(action)
        if(not action):
            if(high_lever):
                num_correct_presses=0
                feedback = int(rand() < .2)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe      
            else:
                feedback = int(rand()< .8)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe 
                if(num_correct_presses>4):
                    high_lever=1
                    num_correct_presses=0


        else:
            if(high_lever):
                feedback = int(rand() < .8)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)
                if(num_correct_presses>4):
                    high_lever=0
                    num_correct_presses=0

            else:
                feedback = int(rand() < .2)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)

    for x in range (trial_length):
        if (x == 0):
            win_shift.append(0)
            win_stay.append(0)
            lose_shift.append(0)
            lose_stay.append(0)

        else:
            if (reward_history[x-1]==1):
                if (action_history[x-1] == action_history[x]):
                    win_shift.append(0)
                    win_stay.append(1)
                    lose_shift.append(0)
                    lose_stay.append(0)
                else:
                    win_shift.append(1)
                    win_stay.append(0)
                    lose_shift.append(0)
                    lose_stay.append(0) 
            else:
                if (action_history[x-1] == action_history[x]):
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(0)
                    lose_stay.append(1)
                else:
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(1)
                    lose_stay.append(0)

    PercentWinStay = (sum(win_stay)/(sum(win_stay)+sum(win_shift)))
    PercentLoseShift = (sum(lose_shift)/(sum(lose_stay)+sum(lose_shift)))

    return action_history,reward_history, PercentWinStay, PercentLoseShift

## Generate data without PRL
def generate_data_NoPRL(params,trial_length):
    [alpha,beta_param] = params
    [q_left, q_right] = [.5, .5]
    action_history=[]
    reward_history=[]
    win_shift=[]
    win_stay=[]
    lose_shift=[]
    lose_stay=[]

    high_lever=1
    for i in range(trial_length):
        p_left = getActionProb(q_left, q_right, beta_param)
        p_right = 1 - p_left
        action = int(rand() < p_left)
        action_history.append(action)
        if(not action):
            if(high_lever):
                feedback = int(rand() < .2)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe      
            else:
                feedback = int(rand()< .8)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe      


        else:
            if(high_lever):
                feedback = int(rand() < .8)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)

            else:
                feedback = int(rand() < .2)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)

    for x in range (trial_length):
        if (x == 0):
            win_shift.append(0)
            win_stay.append(0)
            lose_shift.append(0)
            lose_stay.append(0)

        else:
            if (reward_history[x-1]==1):
                if (action_history[x-1] == action_history[x]):
                    win_shift.append(0)
                    win_stay.append(1)
                    lose_shift.append(0)
                    lose_stay.append(0)
                else:
                    win_shift.append(1)
                    win_stay.append(0)
                    lose_shift.append(0)
                    lose_stay.append(0) 
            else:
                if (action_history[x-1] == action_history[x]):
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(0)
                    lose_stay.append(1)
                else:
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(1)
                    lose_stay.append(0)


    PercentWinStay = (sum(win_stay)/(sum(win_stay)+sum(win_shift)))
    PercentLoseShift = (sum(lose_shift)/(sum(lose_stay)+sum(lose_shift)))

    return action_history,reward_history, PercentWinStay, PercentLoseShift
