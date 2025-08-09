import cv2
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic


class MobileSAMOnnx:
    def __init__(self, onnx_checkpoint, sam_checkpoint, model_type="vit_t", device='cuda'):
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.onnx_model = SamOnnxModel(self.sam, return_single_mask=True)
        self.ort_session = onnxruntime.InferenceSession(onnx_checkpoint)

        self.sam.to(device='cpu')
        self.predictor = SamPredictor(self.sam)

    def show_mask(self, mask, ax=None):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image
        # ax.imshow(mask_image)

    def overlay_mask_on_image(self, image_rgb: np.ndarray, mask: np.ndarray,
                              color_rgb=(30, 144, 255), alpha=0.6) -> np.ndarray:
        """
        image_rgb: (H, W, 3), RGB
        mask: (H, W) bool or {0,1} float
        color_rgb: マスク色 (R,G,B)
        alpha: マスクの透過率
        """
        h, w = image_rgb.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32)  # 0/1
        color = np.array(color_rgb, dtype=np.float32)[None, None, :]  # (1,1,3)

        # アルファブレンド: img*(1 - a*m) + color*(a*m)
        overlay = image_rgb.astype(np.float32) * (1.0 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def draw_input_points(self, image_rgb: np.ndarray, coords_xy: np.ndarray,
                          labels: np.ndarray, marker_size=20, thickness=2) -> np.ndarray:
        """
        coords_xy: [[x, y], ...] 画像座標 (pixel)
        labels: 1=positive(緑の星), 0=negative(赤の星)
        """
        out = image_rgb.copy()
        # OpenCVの星マーカー
        for (x, y), lab in zip(coords_xy.astype(int), labels.astype(int)):
            color = (0, 255, 0) if lab == 1 else (255, 0, 0)  # BGRでもRGBでも(0,255,0)は緑になる
            cv2.drawMarker(out, (int(x), int(y)),
                           color=color,
                           markerType=cv2.MARKER_STAR,
                           markerSize=marker_size,
                           thickness=thickness,
                           line_type=cv2.LINE_AA)
        return out

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def predict(self, image_input, input_point=np.array([[250, 375]]), input_label=np.array([1]), visualize=True, use_rgb=True):
        # 入力が str（パス）の場合は読み込み、np.ndarray の場合はそのまま使用
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:  # グレースケール
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                # BGR → RGB 変換の可能性を考慮（OpenCV画像ならBGR）
                if use_rgb is False:
                    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_input
            else:
                raise ValueError("Unsupported ndarray shape for image input.")
        else:
            raise TypeError("image_input must be a file path (str) or an image array (np.ndarray).")

        self.predictor.set_image(image_rgb)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        print("Image embedding shape:", image_embedding.shape)

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, image_rgb.shape[:2]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image_rgb.shape[:2], dtype=np.float32)
        }

        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)

        # --- ここが重要 ---
        # しきい値適用（masks がロジット/確率ならしきい値を適用）
        thr = getattr(self.predictor.model, "mask_threshold", 0.0)
        masks = (masks > thr) if masks.dtype != np.bool_ else masks

        # 形状正規化: (B, C, H, W) -> (H, W)
        if masks.ndim == 4:
            # C>1 の場合は1枚選ぶ or 結合（例: max）
            if masks.shape[1] > 1:
                masks = masks.max(axis=1, keepdims=True)  # 複数マスク結合（最大）
            masks = masks.squeeze(0).squeeze(0)  # -> (H, W)
        elif masks.ndim == 3:
            masks = masks.squeeze(0)  # -> (H, W)
        # ここまでで masks は (H, W)

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            segment_image = self.show_mask(masks, plt.gca())
            self.show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            plt.show()
            return segment_image, masks, low_res_logits
        else:
            mask_hw = masks.astype(np.uint8)  # (H, W), 0/1
            seg_rgb = self.overlay_mask_on_image(image_rgb, mask_hw, color_rgb=(30, 144, 255), alpha=0.6)
            # 入力点の星マーカー描画（緑=positive、赤=negative）
            seg_with_points = self.draw_input_points(seg_rgb, input_point, input_label, marker_size=20, thickness=2)
            seg_with_points = cv2.cvtColor(seg_with_points, cv2.COLOR_RGB2BGR)
            cv2.imwrite("output_mask.png", seg_with_points)
            cv2.imshow("segmented image", seg_with_points)
            cv2.waitKey(1)
            return seg_with_points, masks, low_res_logits

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

                # 重畳
                seg_rgb = self.overlay_mask_on_image(image_rgb, mask_hw, color_rgb=color_rgb, alpha=alpha)
                # 星マーカー描画
                seg_with_points = self.draw_input_points(seg_rgb, input_point, input_label,
                                                         marker_size=24, thickness=2)
                # 表示（RGB->BGR）
                vis_bgr = cv2.cvtColor(seg_with_points, cv2.COLOR_RGB2BGR)
                cv2.imshow("MobileSAM-ONNX Webcam", vis_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(save_path, vis_bgr)
                    print(f"Saved: {save_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sam_checkpoint = "/home/yuga/usr/yuga_ws/gaze_based_attention/third_party/MobileSAM/weights/mobile_sam.pt"
    onnx_checkpoint = "/home/yuga/usr/yuga_ws/gaze_based_attention/third_party/MobileSAM/mobile_sam.onnx"
    image_path = "/home/yuga/usr/yuga_ws/gaze_based_attention/io/picture2.jpg"
    cls = MobileSAMOnnx(onnx_checkpoint=onnx_checkpoint, sam_checkpoint=sam_checkpoint, model_type="vit_t", device='cuda')
    # cls.predict(image_path=image_path, visualize=False)
    cls.run_webcam()
