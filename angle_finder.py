#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg
#Double check library dependencies needed


class AngleFinder:
    
    def __init__(self):
        rospy.init_node('angle_finder')
        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.needsAngle = True

    def scan_callback(self, msg):
        """
        Made for LiDAR use (with range of values 0-1147) (Hardcoding values is
        simpler)

        360/1147=0.314
        Check: 573 (half of 1147 and therefore 180 degrees) * 0.314 = 179.9
        And, reverse: =573/180=3.18
        
        Now Side 1 (Left side?): 30-150 degrees -> 95.4-477
        so 135*3.18 = 429
        or 150 * 3.18 = 477
        and 30 * 3.18 = 95.4

        Side 2 (Right side?): 210-330 -> 667-1049
        210*3.18 = 667
        667*3.18 = 1049
        """
        if self.needsAngle:
            closest_obj = 30
            closest_idx
            obj_dsts = list(msg.ranges)
            #iterate over one side of scan values
            for idx in range(95, 477):
                
                #find closest object and that index in the object distances list
                if obj_dsts[idx] < closest_obj:
                    closest_obj = obj_dsts[idx]
                    closest_idx = idx

            print("Best angle is {} with distance {}".format(closest_idx, closest_obj))



            

    def run(self):
        rospy.spin()

    if __name__ == '__main__':
        node = AngleFinder()
        node.run()