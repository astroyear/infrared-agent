from pathlib import Path
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = RTDETR(str(ROOT / "pwl_rtdetr" / "best.pt"))
    print("模型类别：", model.names)

    results = model.predict(
        source=str(
            ROOT / "test_images" / "hituav" / "1_60_50_0_07426.jpg"
        ),
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        project=str(ROOT / "runs" / "predict"),
        name="pwl_test",
    )

    for result in results:
        print("检测框数量：", len(result.boxes))
        print("结果保存目录：", result.save_dir)