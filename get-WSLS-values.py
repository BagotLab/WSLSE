import numpy as np 
import pandas as pd
import scipy as sp
from pylab import * 


## Input dataframe with two columns containing binary actions and rewards
def getWSLS(df):
    win_shift=[]
    lose_shift=[]
    win_stay=[]
    lose_stay=[]
    win_shift2=[]
    lose_shift2=[]
    win_stay2=[]
    lose_stay2=[]
    
    ah = df['action'].values
    rh = df['reward'].values
    
    for x in range(len(ah)):
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
            if (rh[x-1]==1):
                if (ah[x-1] == ah[x]):
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
                if (ah[x-1] == ah[x]):
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(0)
                    lose_stay.append(1)
                else:
                    win_shift.append(0)
                    win_stay.append(0)
                    lose_shift.append(1)
                    lose_stay.append(0)
            if (rh[x-2]==1):
                if (ah[x-2] == ah[x]):
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
                if (ah[x-2] == ah[x]):
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
    return PercentWinStay, PercentLoseShift
 
