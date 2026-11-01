import argparse
from pathlib import Path
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 PWL-RTDETR 检测单张红外图片")
    parser.add_argument("image", nargs="?", type=Path,
                        default=ROOT / "test_images" / "hituav" / "1_60_50_0_07426.jpg",
                        help="图片路径，默认使用已验证的 HIT-UAV 样图")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "predict",
                        help="保存各次预测结果的父目录")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"找不到图片文件：{args.image}")

    model = RTDETR(str(ROOT / "pwl_rtdetr" / "best.pt"))
    print("模型类别：", model.names)

    results = model.predict(
        source=str(args.image),
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        project=str(args.output_dir),
        name="pwl_test",
    )

    for result in results:
        print("检测框数量：", len(result.boxes))
        print("结果保存目录：", result.save_dir)
