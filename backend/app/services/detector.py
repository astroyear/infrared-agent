import hashlib
import os
import sys
from pathlib import Path
from threading import Lock

from ..core import ROOT

class Detector:
    def __init__(self):
        self.lock = Lock()

    def load(self):
        source = Path(os.getenv("MODEL_SOURCE_DIR", ROOT / "RTDETR-main-code")).resolve()
        weights = Path(os.getenv("MODEL_PATH", source / "pwl_rtdetr/best.pt"))
        sys.path.insert(0, str(source))
        import ultralytics
        if not Path(ultralytics.__file__).resolve().is_relative_to(source):
            raise RuntimeError("必须使用仓库中的 ultralytics")
        self.model = ultralytics.RTDETR(str(weights))
        with weights.open("rb") as stream:
            self.digest = hashlib.file_digest(stream, "sha256").hexdigest()

    def predict(self, image, dest, conf):
        import cv2
        with self.lock:
            result = self.model.predict(str(image), device=os.getenv("DEVICE", "cpu"),
                                        imgsz=640, conf=conf, save=False, verbose=False)[0]
            rows = result.boxes.data.cpu().tolist()
            ok, jpg = cv2.imencode(".jpg", result.plot())
        if not ok:
            raise IOError("预测图编码失败")
        temporary = dest.with_suffix(".tmp")
        temporary.write_bytes(jpg.tobytes())
        temporary.replace(dest)
        height, width = result.orig_shape
        boxes = [{"class_id": int(cls), "class_name": result.names[int(cls)],
                  "confidence": score, "bbox_xyxy": [x1, y1, x2, y2]}
                 for x1, y1, x2, y2, score, cls in rows]
        return {"width": width, "height": height, "model_hash": self.digest,
                "parameters": {"imgsz": 640, "conf": conf}, "detections": boxes}
