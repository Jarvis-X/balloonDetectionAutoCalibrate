from __future__ import print_function
import pixy
from ctypes import *
from pixy import *

import rospy
from std_msgs.msg import Float64MultiArray

rospy.init_node("detection", anonymous=True)
rate = rospy.Rate(5)
balloonpub = rospy.Publisher('balloon', Float64MultiArray, queue_size=1)
squarepub = rospy.Publisher('square', Float64MultiArray, queue_size=1)

# Pixy2 Python SWIG get blocks example #

print("Pixy2 Python SWIG Example -- Get Blocks")

pixy.init ()
pixy.change_prog ("color_connected_components");

class Blocks (Structure):
  _fields_ = [ ("m_signature", c_uint),
    ("m_x", c_uint),
    ("m_y", c_uint),
    ("m_width", c_uint),
    ("m_height", c_uint),
    ("m_angle", c_uint),
    ("m_index", c_uint),
    ("m_age", c_uint) ]

blocks = BlockArray(100)
balloonmsg = Float64MultiArray()
squaremsg = Float64MultiArray()
balloonmsg.data = [0, 0, 0, 0]
squaremsg.data = [0, 0, 0, 0]

while not rospy.is_shutdown():
  rate.sleep()
  count = pixy.ccc_get_blocks(100, blocks)
  if count > 0:
    for index in range (0, count):
      if blocks[index].m_signature == 1:
        balloonmsg.data[0] = 1
        send = max(blocks[index].m_width, blocks[index].m_height)
        print('[BLUE BALLOON: X=%3d Y=%3d MAX=%3d]' % (blocks[index].m_x, blocks[index].m_y, send))
        balloonmsg.data[1] = blocks[index].m_x
        balloonmsg.data[2] = blocks[index].m_y
        balloonmsg.data[3] = send
      elif blocks[index].m_signature == 2:
        squaremsg.data[0] = 1
        send = max(blocks[index].m_width, blocks[index].m_height)
        print('[YELLOW SQUARE GOAL: X=%3d Y=%3d MAX=%3d]' % (blocks[index].m_x, blocks[index].m_y, send))
        squaremsg.data[0] = blocks[index].m_x
        squaremsg.data[0] = blocks[index].m_y
        squaremsg.data[0] = send
  else:
    balloonmsg.data[0] = 0
    squaremsg.data[0] = 0
  balloonpub.publish(balloonmsg)
  squarepub.publish(squaremsg)
  
  