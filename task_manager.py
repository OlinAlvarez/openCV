import detect_dice
import detect_gate
import detect_roulette
import detect_cashIn
import detect_buoy
'''
This is the class which will handle all the tasks.
Gate,Dice,Roulette,CashIn
'''


class TaskManager():
    def __init__(self,taskName):
        self.taskName = taskName.lower()
        self.task
        if taskName == 'gate':
            self.task = self.gateDetect()
        elif taskName == 'dice':
            self.task = self.diceDetect()
        elif taskName == 'roulette':
            self.task = self.rouletteDetect()
        elif taskName == 'cashin':
            self.task = self.cashInDetect()
        elif taskName == 'buoy':
            self.task = self.buoyDetect()
        else:
            print 'Error task not found'
    def gateDetect(self):
        return 0
    def diceDetect(self):
        return 0
    def rouletteDetect(self):
        return 0
    def cashInDetect(self):
        return 0
    def buoyDetect(self):
        return detect_buoy.detect_buoy()