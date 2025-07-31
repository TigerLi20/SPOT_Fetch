##1. evaluation process:
1. process videos:  "python3 gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/2m_data.csv"
2. data clean up: "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/2m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --threshold 0.4"
3. evaluate cleaned data using csv and the original video: --> pointing_eval.py
4. visualize evaluation results


##2. frame transformation:
1. April tag(id=2, width = 16.5 cm) at fixed location on the ground
2. compute the location from camera center and find the transformation matrix to transform px values to meter
3. use pose landmark 2d estimation to find pixel value of hip center and use depth image to find the average depth of the hip center(i.e. where the human is)
4. apply the same transformation to hip center [px, px, m] --> [m, m, m]
(3.5). align april tag plane to have directional vector of (0, 1, 0) and align hip center
5. rotate and translate all of the pose landmark 3d estimation point based on transformed 3d hip center

2d pose landmark:
    x(+) -> right
    y(+) -> down
    z(+) -> away from camera, zero at hip(i.e. negative in front of the human). similar scale as x

3d pose landmark (hip as origin [0, 0, 0]):
    x(+) -> right
    y(+) -> down
    z(+) -> away from camera