# Spot+GENMOS System Setup
[robotdev setup reference guide](https://github.com/h2r/robotdev/tree/v0.3.1)

[genmos+spot setup reference guide](https://github.com/zkytony/genmos_object_search/wiki/100-GenMOS:-A-System-for-Generalized-Multi-Object-Search)
1. Clone this repo and setup docker if haven't. 
2. Download [lab121_rtabmap.db](https://drive.google.com/file/d/1_JMnKoypx9JzqWXAsdzw1yA8wgltem-z/view?usp=drive_link) and add it to ```/robotdev/spot/ros_ws/src/rbd_spot_perception/maps/rtabmap/```

4. build the docker image for spot
```
cd robotdev
source docker/build.noetic.sh --gui(Optional)
```
```
# to start running docker
source docker/run.noetic.sh
```
3. go to ```robotdev/genmos_object_search``` folder, follow [GENMOS Github](https://github.com/zkytony/genmos_object_search/wiki/100-GenMOS:-A-System-for-Generalized-Multi-Object-Search) to install genmos.
4. go to ```robotdev/spot/ros_ws/genmos_object_search_ros/``` to install dependencies for GenMOS_ros
```cd /robotdev/spot/ros/src/genmos_object_search_ros/
source install_dependencies.bash
```
5. Setup the spot workspace
```
# at robotdev root directory:
source docker/run.noetic.sh
cd repo/robotdev
source setup_spot.bash
```
6. run ```build_spot``` to ensure that the package is built successfully. 
### debug if needed

7. check to make sure protobuf version is correct
``` pip install protobuf==3.20.0 ```
8. change local_cloud_publisher node access: go to ```ros/src/genmos_ros/processing/local_cloud_publisher.cpp``` to change the read write access of the local_cloud_publisher ```chmod +x local_cloud_publisher.cpp```


### Verify that the test in simulation works
To run the test, do the following IN ORDER:
(create a .screenrc for future simplicity)
0. run ```python config_simple_sim_lab121_lidar.py``` to generate the .yaml configuration file, this file is located in:
```/robotdev/spot/ros_ws/src/genmos_object_search_ros/tests/spot/grpc/simple_sim```

1. run in a terminal
2. ```
   roslaunch rbd_spot_perception graphnav_map_publisher.launch map_name:=cit121
   ```
4. run in a terminal
   ```
   roslaunch genmos_object_search_ros spot_local_cloud_publisher.launch robot_pose_topic:=/simple_sim_env/init_robot_pose
   ```
6. run in a terminal
   ```
   roslaunch genmos_object_search_ros simple_sim_env.launch map_name:=cit121
   ```
9. run in a terminal
    ```
    python -m genmos_object_search.grpc.server
    ```
12. run in a terminal
    ```
    python test_simple_sim_env_local_search_3d.py groundtruth 32 config_simple_sim_lab121_lidar.yaml
    ```
14. run in a terminal
    ```
    roslaunch genmos_object_search_ros view_simple_sim.launch
    ```
16. to monitor CPU temperature:
    ```
    watch -n 1 -x sensors
    ```


# Create then activate the virtual environment
```
conda install -n legs python=3.10
conda activate legs
```

# Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers
```
