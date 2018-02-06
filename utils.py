import math
def printConfusionMatrix(result, labels):
    result0 = [int(x) for x in result]

    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0

    for i, row in enumerate(labels):
        if row  == 1 and result[i] == 1:
            truePositives +=1
        if row  != 1 and result[i] != 1:
            trueNegatives +=1
        if row  != 1 and result[i] == 1:
            falsePositives +=1
        if row == 1 and result[i] != 1:
            falseNegatives +=1

    print "true pos:", truePositives, "true neg:", trueNegatives, "false pos:", falsePositives, "false neg:", falseNegatives, "\n"


def find_angle(center,x,y,w,h):
    midpoint = ( x + ( w / 2), y + ( h / 2))
    angle = math.atan2( midpoint[1] - center[1], midpoint[0] - center[0])
    return angle

GREEN = 0
BLACK = 1
MAGENTA = 2
RED = 3
GOLDEN_YELLOW = 4
WHITE = 5
BLUE = 6
colors = [(0,255,0),(0,0,0),(255,0,255),(255,0,0),(255,165,0),(255,255,255),(0,0,255)]
def get_direction(x,y,w,h):
    print 'in get directions'
    wPad = w / 3
    hPad = h / 3
    cx = center[0]
    cy = center[1]
    print cx, cy, x , y
    print 'left bound', x + wPad
    print 'right bound', x + (2 * wPad)
    if(cx < x + wPad):
        if(cx > x + (2 * wPad)):
            direction[0] = 0
        else:
            direction[0] = 1
    else:
        direction[0] = -1
    print 'up bound', y + wPad
    print 'down bound', y + (2 * wPad)
    if(cy > y + hPad):
        if(cy < y + (2 * hPad)):
            direction[1] = 0
        else:
            direction[1] = 1
    else:
        direction[1] = -1
