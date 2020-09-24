import numpy as np 
import pandas as pd
import scipy as sp
from pylab import * 

def getActionProb(q1, q2,  beta):
    return exp(beta*q1)/( exp(beta*q1) + exp(beta*q2))

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
    win_stay2=[]
    lose_shift2=[]
    win_shift2=[]
    lose_stay2=[]
    #num_correct_presses=0
    prev_correct=False
    high_lever=1
    for i in range(trial_length):
        p_left = getActionProb(q_left, q_right, beta_param)
        p_right = 1 - p_left
        action = int(rand() < p_left)
        action_history.append(action)
        if(not action):
            if(high_lever):
                #num_correct_presses = 0
                feedback = int(rand() < .2)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe      
            else:
                feedback = int(rand()< .8)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe      
                #num_correct_presses=num_correct_presses+1
                #if(num_correct_presses>4):
                 #   high_lever=1
                  #  num_correct_presses=0

        else:
            if(high_lever):
                feedback = int(rand() < .8)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)

            else:
                #num_correct_presses = 0
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
            win_shift2.append(0)
            win_stay2.append(0)
            lose_shift2.append(0)
            lose_stay2.append(0)
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
            if (reward_history[x-2]==1):
                if (action_history[x-2] == action_history[x]):
                    win_shift2.append(0)
                    win_stay2.append(1)
                    lose_shift2.append(0)
                    lose_stay2.append(0)
                else:
                    win_shift2.append(1)
                    win_stay2.append(0)
                    lose_shift2.append(0)
                    lose_stay2.append(0) 
            else:
                if (action_history[x-2] == action_history[x]):
                    win_shift2.append(0)
                    win_stay2.append(0)
                    lose_shift2.append(0)
                    lose_stay2.append(1)
                else:
                    win_shift2.append(0)
                    win_stay2.append(0)
                    lose_shift2.append(1)
                    lose_stay2.append(0)
      


    PercentWinStay = (sum(win_stay)/(sum(win_stay)+sum(win_shift)))
    PercentLoseShift = (sum(lose_shift)/(sum(lose_stay)+sum(lose_shift)))
    PercentWinStay2 = (sum(win_stay2)/(sum(win_stay2)+sum(win_shift2)))
    PercentLoseShift2 = (sum(lose_shift2)/(sum(lose_stay2)+sum(lose_shift2)))
    return action_history,reward_history, PercentWinStay, PercentLoseShift,PercentWinStay2,PercentLoseShift2

## Generate data with PRL
def generate_data_PRL(params,trial_length):
    [alpha,beta_param] = params
    [q_left, q_right] = [.5, .5]
    action_history=[]
    reward_history=[]
    win_shift=[]
    win_stay=[]
    lose_shift=[]
    lose_stay=[]
    win_stay2=[]
    lose_shift2=[]
    win_shift2=[]
    lose_stay2=[]
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
                feedback = int(rand() < .8)
                reward_history.append(feedback)
                pe = (feedback - q_right)
                q_right = q_right + alpha*pe
                num_correct_presses=num_correct_presses+1
                if(num_correct_presses>4):
                    high_lever=1
                    num_correct_presses=0
        else:
            if(high_lever):
                feedback = int(rand() < .8)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)
                num_correct_presses=num_correct_presses+1
                if(num_correct_presses>4):
                    high_lever=0
                    num_correct_presses=0
            else:
                feedback = int(rand() < .2)
                pe = (feedback - q_left)  
                q_left = q_left + alpha*pe
                reward_history.append(feedback)

    for x in range (trial_length):
        if (x == 0): # if first trial - no stay/shift behavior yet
            win_shift.append(0)
            win_stay.append(0)
            lose_shift.append(0)
            lose_stay.append(0)
            win_shift2.append(0)
            win_stay2.append(0)
            lose_shift2.append(0)
            lose_stay2.append(0)
            
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
            if (reward_history[x-2]==1):
                if (action_history[x-2] == action_history[x]):
                    win_shift2.append(0)
                    win_stay2.append(1)
                    lose_shift2.append(0)
                    lose_stay2.append(0)
                else:
                    win_shift2.append(1)
                    win_stay2.append(0)
                    lose_shift2.append(0)
                    lose_stay2.append(0) 
            else:
                if (action_history[x-2] == action_history[x]):
                    win_shift2.append(0)
                    win_stay2.append(0)
                    lose_shift2.append(0)
                    lose_stay2.append(1)
                else:
                    win_shift2.append(0)
                    win_stay2.append(0)
                    lose_shift2.append(1)
                    lose_stay2.append(0)
    PercentWinStay = (sum(win_stay)/(sum(win_stay)+sum(win_shift)))
    PercentLoseShift = (sum(lose_shift)/(sum(lose_stay)+sum(lose_shift)))
    PercentWinStay2 = (sum(win_stay2)/(sum(win_stay2)+sum(win_shift2)))
    PercentLoseShift2 = (sum(lose_shift2)/(sum(lose_stay2)+sum(lose_shift2)))

    return action_history,reward_history, PercentWinStay, PercentLoseShift,PercentWinStay2,PercentLoseShift2

def create_numpy_arrays():
    means_wsr=np.zeros((101,11))
    means_lsr=np.zeros((101,11))
    means_wsr2=np.zeros((101,11))
    means_lsr2=np.zeros((101,11))    
    covariances=np.zeros((101,11,101,11,2,2))
    for j in range(0,101): #iterate over every .01 increment of learning rate from 0 to 1 
        for i in range (0,11): #iterate over every 1 increment of inverse temperature from 0 to 10
            alpha = j*.01
            beta = i 
            wsr = []
            lsr = []
            wsr2 = []
            lsr2 = []
            
        for p in range(1001): #1000 samples per param combination
            ah,rh,ws,ls,ws2,lse = generate_data_PRL([alpha, beta],100)
            wsr.append(ws)
            lsr.append(ls)
            wsr2.append(ws2)
            lsr2.append(ls2)            
            means_wsr[j,i]=np.mean(wsr)
            means_lsr[j,i]=np.mean(lsr)
            means_wsr2[j,i]=np.mean(wsr2)
            means_lsr2[j,i]=np.mean(lsr2)            
            covariances[j,i]=np.cov(wsr,lsr,wsr2,lsr2)
            
    np.save("means_wsr",means_wsr)
    np.save("means_lsr",means_lsr)  
    np.save("means_wsr2",means_wsr2)
    np.save("means_lsr2",means_lsr2)     
    np.save("covariances",covariances) 
