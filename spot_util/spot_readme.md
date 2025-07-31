# change the resolution of the gripper camera to have the highest resolution
# other options: ['640x480', '1280x720', '1920x1080', '3840x2160', '4096x2160','4208x3120']

python set_gripper_cam_param.py 138.16.161.21 --resolution 4208x3120

 --resolution 4208x3120
"""
focal_length {
  x: 3629.5913404959015
  y: 3629.5913404959015
}
principal_point {
  x: 2104.0
  y: 1560.0
}

"""

 --resolution '640x480'
"""
focal_length {
  x: 552.0291012161067
  y: 552.0291012161067
}
principal_point {
  x: 320.0
  y: 240.0
}
"""
# collect images  using spot_image_capture

# REMEMBER to change the gripper camera resolution back to '640x480' after using. 

python set_gripper_cam_param.py 138.16.161.21 --resolution 640x480

