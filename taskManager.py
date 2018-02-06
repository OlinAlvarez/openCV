import detect_dice
import detect_gate
import detect_roulette
import detect_cashIn
'''
This is the class which will handle all the tasks.
Gate,Dice,Roulette,CashIn
'''
class TaskManager():
    def __init__(self,task):
        task = task.lower()
        if(task == 'gate'):
            self.gateDetect()
        elif(task == 'dice'):
            self.diceDetect()
        elif(task == 'roulette'):
            self.rouletteDetect()
        elif(task == 'cashin'):
            self.cashInDetect()
        else:
            print 'Error task not found'
    def gateDetect(self):
        return
    def diceDetect(self):
        return
    def rouletteDetect(self):
        return
    def cashInDetect(self):
        return