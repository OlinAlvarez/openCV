import task_manager as tm
import detect_buoy as db
bd = db.detect_buoy()
print bd.coords
while not bd.isTaskComplete:
    bd.detect()
    '''
    Update this to be a custom ROS message.
    the information must be of the form
    (x,y)
    where x,y are in {-1,0,1}
     x
    -1 left
     0 steady
     1 right
     
     y
     -1 down
     0 steady
     1 up
    '''
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
