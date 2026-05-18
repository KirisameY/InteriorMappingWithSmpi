
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试 EXR 文件信息")
    parser.add_argument("-e", "--exr_path", type=str, required=True, help="EXR 文件路径")
    args = parser.parse_args()

    import OpenEXR
    exr_file = OpenEXR.InputFile(args.exr_path)
    header = exr_file.header()
    print("EXR Header Information:")
    for key, value in header.items():
        print(f"{key}: {value}")