import pyrealsense2 as rs


RealSenseFilters = {
    "decimation": rs.decimation_filter,
    "spatial": rs.spatial_filter,
    "temporal": rs.temporal_filter,
    "hole-filling": rs.hole_filling_filter
}
