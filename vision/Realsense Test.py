import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    15
)

print("Starting...")

pipeline.start(config)

print("Started!")

try:
    for i in range(100):
        frames = pipeline.wait_for_frames(10000)
        depth = frames.get_depth_frame()

        if depth:
            print("Got frame:", i)

finally:
    pipeline.stop()