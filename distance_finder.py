'''
All lengths must be set in mm for this to work
'''
#THIS IS THE FOCAL LENGTH FOR MAIN CAMERAS
focal_length =  1.37  #in mms
'''
Test values using a red notebook
'''
known_width = 190 #19 cm
known_height = 247 #24.7 cm

def get_distance(perceived_width):
    return (known_width * focal_length) / perceived_width
