import os
import sys
import cv2
import torch
import logging
import warnings
import traceback
import threading
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from torchvision.transforms.functional import to_tensor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


class MobilesamCutieTracking:
    def __init__(self, onnx_checkpoint, sam_checkpoint, model_type="vit_t", device='cuda'):
        self.device = device
        self.is_tracking = False
        self.use_cv2_window = False
        self.run_enable = True
        self.tracking_thread = None
        self.target_pil_rgb = None

        self.lock = threading.Lock()

        # cutie model
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = 480

        # Load MobileSAM model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.onnx_model = SamOnnxModel(self.sam, return_single_mask=True)
        self.ort_session = onnxruntime.InferenceSession(onnx_checkpoint)

        self.sam.to(device='cpu')
        self.predictor = SamPredictor(self.sam)

    @staticmethod
    def cv2_to_pillow_mask(mask):
        arr = mask.detach().cpu().numpy()

        # 形状を (H, W) にする
        if arr.ndim == 3:
            # (1, H, W) -> (H, W)
            if arr.shape[0] == 1:
                arr = arr[0]
            # (H, W, 1) -> (H, W)
            elif arr.shape[2] == 1:
                arr = arr[..., 0]
            else:
                # チャンネル次元が複数ある場合は最大スコア/先頭などに集約
                # ここでは先頭チャンネルを使用（必要なら argmax に変更）
                arr = arr[0]

        # dtype 整形（bool -> 0/255、float -> 0/255）
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8) * 255
        else:
            # 値域が [0,1] なら 0~255 へ
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        return PILImage.fromarray(arr, mode="L")

    @staticmethod
    def cv2_to_pillow_image(cv_image: np.ndarray) -> PILImage:
        if len(cv_image.shape) == 2:
            return PILImage.fromarray(cv_image, mode='L')
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(rgb_image)
        else:
            raise ValueError("Unsupport Image type")

    @staticmethod
    def pillow_to_cv2(pil_img: PILImage) -> np.ndarray:
        """Convert Pillow image to OpenCV BGR numpy array"""
        img_np = np.array(pil_img)
        if pil_img.mode == "RGBA":
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pil_img.mode == "RGB":
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif pil_img.mode == "L":
            return img_np
        else:
            raise ValueError(f"Unsupported image mode: {pil_img.mode}")

    def make_overlay_image(self, pillow_mask, pillow_bgr, color=(0, 0, 255), alpha=0.5):
        cv_bgr = self.pillow_to_cv2(pillow_bgr)
        cv_mask = self.pillow_to_cv2(pillow_mask)

        if len(cv_mask.shape) == 3 and cv_mask.shape[2] == 3:
            cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
        else:
            cv_mask = cv_mask

        # マスクを2値化
        mask_bin = (cv_mask > 0).astype(np.uint8)

        # カラーマスクを作成
        color_mask = np.zeros_like(cv_bgr)
        color_mask[:] = color

        # マスク領域だけ色を合成
        overlay = cv_bgr.copy()
        overlay[mask_bin == 1] = cv2.addWeighted(
            cv_bgr[mask_bin == 1], 1 - alpha,
            color_mask[mask_bin == 1], alpha,
            0
        )

        return overlay

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def run_webcam(self, camera_index=0, alpha=0.6, color_rgb=(30, 144, 255),
                   default_point="center", save_path="webcam_seg.png"):
        """
        Webカメラから連続推論して表示。
        - 左クリック: 入力点を更新（positive=緑の星）
        - sキー: 現在のフレームを保存
        - qキー: 終了
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index}")

        # マウスクリックで入力点を動かせるようにする
        state = {"point": None}  # (x, y) in pixel on current frame

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["point"] = (x, y)  # positive point
                self.is_tracking = False

        cv2.namedWindow("MobileSAM-ONNX Webcam")
        cv2.setMouseCallback("MobileSAM-ONNX Webcam", on_mouse)

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("Frame grab failed")
                    break

                # BGR -> RGB
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                target_pil_bgr = PILImage.fromarray(frame_bgr)
                H, W = image_rgb.shape[:2]

                # 入力点の決定
                if state["point"] is None:
                    if default_point == "center":
                        input_point = np.array([[W // 2, H // 2]], dtype=np.float32)
                    else:
                        # 画面左上25%の例（必要ならカスタムしてください）
                        input_point = np.array([[int(W * 0.25), int(H * 0.25)]], dtype=np.float32)
                else:
                    input_point = np.array([[state["point"][0], state["point"][1]]], dtype=np.float32)

                input_label = np.array([1], dtype=np.float32)  # positive

                # SAM predictor で埋め込み
                if self.is_tracking is False:
                    self.processor.clear_memory()
                    self.predictor.set_image(image_rgb)
                    image_embedding = self.predictor.get_image_embedding().cpu().numpy()

                    # ONNX 入力に座標・ラベル変換
                    onnx_coord = np.concatenate(
                        [input_point, np.array([[0.0, 0.0]], dtype=np.float32)],
                        axis=0
                    )[None, :, :]
                    onnx_label = np.concatenate(
                        [input_label, np.array([-1], dtype=np.float32)],
                        axis=0
                    )[None, :].astype(np.float32)
                    onnx_coord = self.predictor.transform.apply_coords(onnx_coord, image_rgb.shape[:2]).astype(np.float32)

                    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                    ort_inputs = {
                        "image_embeddings": image_embedding,
                        "point_coords": onnx_coord,
                        "point_labels": onnx_label,
                        "mask_input": onnx_mask_input,
                        "has_mask_input": onnx_has_mask_input,
                        "orig_im_size": np.array([H, W], dtype=np.float32),
                    }

                    masks, _, _ = self.ort_session.run(None, ort_inputs)

                    # しきい値適用 & 形状整形
                    thr = getattr(self.predictor.model, "mask_threshold", 0.0)
                    masks = (masks > thr) if masks.dtype != np.bool_ else masks
                    if masks.ndim == 4:
                        if masks.shape[1] > 1:
                            masks = masks.max(axis=1, keepdims=True)
                        masks = masks.squeeze(0).squeeze(0)
                    elif masks.ndim == 3:
                        masks = masks.squeeze(0)
                    mask_hw = masks.astype(np.uint8)  # (H, W)

                    pil_rgb = PILImage.fromarray(image_rgb)
                    pil_mask = PILImage.fromarray(mask_hw, mode="L")

                    # palette is for visualization
                    self.palette = pil_mask.getpalette()
                    self.objects = np.unique(np.array(pil_mask))
                    self.objects = self.objects[self.objects != 0].tolist()

                    torch_rgb = to_tensor(pil_rgb).cuda().float()
                    torch_mask = torch.from_numpy(np.array(pil_mask)).cuda()

                    mask = self.processor.step(torch_rgb, torch_mask, objects=self.objects)
                    self.is_tracking = True

                else:
                    target_image = to_tensor(image_rgb).cuda().float()
                    output_prob = self.processor.step(target_image)  # inference
                    mask = self.processor.output_prob_to_mask(output_prob)

                try:
                    estimation_mask = self.cv2_to_pillow_mask(mask)
                    pil_mask_rgb = estimation_mask.convert("RGB")
                    overlay_mask = self.make_overlay_image(pillow_mask=pil_mask_rgb, pillow_bgr=target_pil_bgr)

                    vis_bgr = cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR)
                    cv2.imshow("MobileSAM-ONNX Webcam", vis_bgr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite(save_path, vis_bgr)
                        print(f"Saved: {save_path}")

                except Exception as e:
                    cv2.imshow("MobileSAM-ONNX Webcam", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    traceback.print_exc()

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sam_checkpoint = "/home/yuga/usr/yuga_ws/gaze_based_attention/third_party/MobileSAM/weights/mobile_sam.pt"
    onnx_checkpoint = "/home/yuga/usr/yuga_ws/gaze_based_attention/third_party/MobileSAM/mobile_sam.onnx"
    image_path = "/home/yuga/usr/yuga_ws/gaze_based_attention/io/picture2.jpg"
    cls = MobilesamCutieTracking(onnx_checkpoint=onnx_checkpoint, sam_checkpoint=sam_checkpoint, model_type="vit_t", device='cuda')
    cls.run_webcam()
