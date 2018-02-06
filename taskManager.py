import detect_dice
import detect_gate
import detect_roulette
import detect_cashIn
'''
This is the class which will handle all the tasks.
Gate,Dice,Roulette,CashIn
'''
def __init__(self,task):
    task = task.lower()
    if(task == 'gate'):
        gateDetect()
    elif(task == 'dice'):
        diceDetect()
    elif(task == 'roulette'):
        rouletteDetect()
    elif(task == 'cashin'):
        cashInDetect()
    else:
        print 'Error task not found'
def gateDetect():
    return
def diceDetect():
    return
def rouletteDetect():
    return
def cashInDetect():
    return