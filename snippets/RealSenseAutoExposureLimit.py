import pyrealsense2 as rs

ctx: rs.context = rs.context()
device: rs.device = ctx.devices[0]

depth_sensor: rs.depth_sensor = device.first_depth_sensor()

# works because it is in range
depth_sensor.set_option(rs.option.auto_exposure_limit, 500.0)

# does not work and results in a wrong error message
depth_sensor.set_option(rs.option.auto_exposure_limit, 0)
