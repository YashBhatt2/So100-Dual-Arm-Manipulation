import pyrealsense2 as rs
import numpy as np
import cv2



# SETTINGS

MARKER_SIZE = 0.05  # metres

# REALSENSE SETUP


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    15
)

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    15
)

profile = pipeline.start(config)



# CAMERA INTRINSICS


color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

intrinsics = color_profile.get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(
    intrinsics.coeffs[:5],
    dtype=np.float64
)



# ARUCO SETUP


aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)



# MAIN LOOP

try:

    while True:

        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(
            color_frame.get_data()
        )

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:

            ids = ids.flatten()

            print("Detected IDs:", ids)

            # Draw all detected markers
            cv2.aruco.drawDetectedMarkers(
                frame,
                corners,
                ids.reshape(-1, 1)
            )

            for i in range(len(ids)):

                print("Processing marker:", ids[i])

                # Only use marker ID 0
                if ids[i] != 0:
                    continue

                print("MARKER 0 FOUND!")

                # Estimate marker pose
                half = MARKER_SIZE / 2

                object_points = np.array([
                    [-half,  half, 0],
                    [ half,  half, 0],
                    [ half, -half, 0],
                    [-half, -half, 0]
                ], dtype=np.float32)

                # 2D coordinates detected in the camera image
                image_points = corners[i][0].astype(np.float32)

                # Estimate marker pose
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )

                if not success:
                    continue

                print("Position:", tvec)

                # Draw XYZ coordinate axes
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    MARKER_SIZE * 0.5
                )

                # Position
                x = float(tvec[0][0])
                y = float(tvec[1][0])
                z = float(tvec[2][0])       

                cv2.putText(
                    frame,
                    "ID: 0",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"X: {x:.3f} m",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Y: {y:.3f} m",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Z: {z:.3f} m",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "SO-100 Object Coordinate Frame",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()