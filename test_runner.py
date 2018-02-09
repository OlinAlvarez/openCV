import taskManager as tm
import detect_buoy as db
bd = db.detect_buoy()
print bd.coords
while not bd.isTaskComplete:
    bd.detect()
    print bd.coords

    '''
    bd = tm.taskManager('buoy')
    print bd
    coords = (0,0)
    while not bd.isTaskComplete():
        bd.detect()
        coords = bd.coords
        print coords
    '''
