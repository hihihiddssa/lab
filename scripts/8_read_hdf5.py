import h5py
import numpy as np

def read_hdf5_file(file_path):
    """
    读取HDF5文件的函数，专注于显示images数据
    参数:
        file_path: HDF5文件的路径
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 打印文件结构
            print("\n=== 文件结构 ===")
            print_structure(f)
            
            # 专门读取images数据集
            if 'images' in f:
                images_group = f['images']
                print("\n=== Images组信息 ===")
                
                for camera_name in images_group.keys():
                    dataset = images_group[camera_name]
                    print(f"\n相机 '{camera_name}':")
                    print(f"形状: {dataset.shape}")
                    print(f"数据类型: {dataset.dtype}")
                    
                    # 显示第一帧的数据示例
                    print(f"第一帧数据示例:")
                    first_frame = dataset[0]
                    print(f"帧大小: {first_frame.shape}")
                    print(f"数值范围: [{np.min(first_frame)}, {np.max(first_frame)}]")
                    print(f"平均值: {np.mean(first_frame)}")
                    
                    # 显示数据集的属性
                    if len(dataset.attrs) > 0:
                        print("属性:")
                        for attr_name, value in dataset.attrs.items():
                            print(f"  {attr_name}: {value}")
            else:
                print("\n文件中没有找到'images'组")
                
    except Exception as e:
        print(f"读取HDF5文件时发生错误: {e}")

def print_structure(obj, level=0):
    """
    递归打印HDF5文件的结构
    """
    indent = "  " * level
    for key in obj.keys():
        if isinstance(obj[key], h5py.Group):
            print(f"{indent}组: {key}/")
            print_structure(obj[key], level + 1)
        else:
            print(f"{indent}数据集: {key} (形状: {obj[key].shape}, 类型: {obj[key].dtype})")

if __name__ == "__main__":
    # 使用示例
    file_path = "/home/asdfminer/YidongYingPan/XtrainerCollectedData/datasets/1104_collect100_yellow_plasticbag_M_notop/train_data/episode_init_11.hdf5"
    read_hdf5_file(file_path)
