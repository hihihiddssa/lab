import pyrealsense2 as rs

# 创建上下文对象
ctx = rs.context()

# 获取所有连接的设备
devices = ctx.query_devices()

# 打印设备信息
print(f"检测到 {len(list(devices))} 个RealSense相机")
for dev in devices:
    print(f"设备序列号: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"设备名称: {dev.get_info(rs.camera_info.name)}")
    print(f"固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
    print("---") 