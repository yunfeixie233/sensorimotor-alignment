from __future__ import annotations

import os
import io
import json
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Iterable,
    Any,
    Callable,
    Optional,
    Sequence,
)
from scipy.spatial.transform import Rotation as R
import transforms3d as t3d
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.utils import image_transform_tensor
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def quantize_state(values, bins: int = 2048) -> List[int]:
    """Map [-1,1] values to integer tokens in [0, bins-1],input MAST be 1-D"""
    tokens = []
    for v in values:
        v = max(-1.0, min(1.0, float(v)))
        idx = int(round((v + 1) / 2 * (bins - 1)))
        tokens.append(idx)
    return tokens


def quantize_action(actions: torch.Tensor, bins: int = 2048) -> torch.Tensor:
    actions = actions.to(torch.float32).clamp(-1.0, 1.0)
    return torch.round((actions + 1) / 2 * (bins - 1)).to(torch.long)


@dataclass
class EpisodeMeta:
    path: str
    T: int
    # For each logical camera (e.g., 'head_camera'), map to the *actual* name in HDF5 (e.g., 'front_camera')
    camera_names: Dict[str, Optional[str]]
    # Storage hints per logical camera: 'array' | 'encoded' (missing cameras omitted)
    camera_storage: Dict[str, str]


class EpisodesHDF5Dataset(Dataset):
    """
    Episodes dataset that pairs HDF5 files in `data_dir` with equally named JSON
    instruction files in `instructions_dir`. Each HDF5 file is treated as one
    episode; frames are flattened across episodes.
    """

    def __init__(
        self,
        dataset_dir: str,
        vocab_offset: int,
        chunk_size: int = 8,
        action_vocab_size=1024,
        resolution: int = 256,
        action_dim: int = 16,
        cameras: Iterable[str] = ("head_camera", "left_camera", "right_camera"),
        image_transform: Optional[Any] = image_transform_tensor,
        float_dtype: torch.dtype = torch.float32,
        preload_numeric: bool = True,
        rotation_type: str = "euler",
        preload_images: bool | str = False,  # False | True | 'meta'
        use_swmr: bool = False,
        frequency_ratio: int = 1,
        original_dataset_len: bool = True,
        action_type: str = "ee",
        predict_image: bool = False,
        predict_text: bool = False,
        allow_missing_cameras: bool = True,
        is_clean_data: bool = False,
        select_num=None,
        jump_num=None,
        input_joint_norm=False,
        joint_action_max=None,
        joint_action_min=None,
        camera_alias: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = (
            os.path.join(dataset_dir, "clean_data")
            if is_clean_data
            else os.path.join(dataset_dir, "data")
        )
        self.instructions_dir = os.path.join(dataset_dir, "instructions")
        self.vocab_offset = vocab_offset
        self.action_type = action_type
        self.chunk_size = chunk_size
        self.action_vocab_size = action_vocab_size
        self.action_dim = action_dim
        self.rotation_type = rotation_type
        self.resolution = resolution
        self.cameras = tuple(cameras)
        self.image_transform = image_transform
        self.float_dtype = float_dtype
        self.select_num = select_num
        self.jump_num = jump_num
        self.preload_numeric = preload_numeric
        self.input_joint_norm = input_joint_norm
        self.joint_action_max = joint_action_max
        self.joint_action_min = joint_action_min
        self.frequency_ratio = frequency_ratio
        self.original_dataset_len = original_dataset_len
        self.predict_image = predict_image
        self.predict_text = predict_text
        self.preload_images = preload_images
        self.use_swmr = use_swmr
        self.allow_missing_cameras = allow_missing_cameras
        # Default aliases: try the same name first, then common alternatives
        self.camera_alias = camera_alias or {
            "head_camera": ("head_camera", "front_camera", "head"),
            "left_camera": ("left_camera", "left"),
            "right_camera": ("right_camera", "right"),
        }

        # Flat index of (episode_idx, t)
        self._index: List[Tuple[int, int]] = []
        # Episode metadata and optional RAM caches
        self._episodes: List[EpisodeMeta] = []
        self._instr_cache: List[Dict[str, List[str]]] = []
        # Optional in-RAM numeric caches (one per episode)
        self._ram_numeric: List[Optional[Dict[str, np.ndarray]]] = []
        # Optional in-RAM image caches (one per episode), keyed by logical camera
        self._ram_images: List[Optional[Dict[str, Any]]] = []
        # Precomputed future indices per episode: List[np.ndarray (T,H)]
        self._future_tables: List[np.ndarray] = []

        # Scan episodes once, build index and (optional) caches
        h5_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".hdf5")]
        h5_files.sort(key=self._natural_key)
        if self.select_num is not None:
            start = self.jump_num if self.jump_num is not None else 0
            end = start + self.select_num
            h5_files = h5_files[start:end]
        if not h5_files:
            raise FileNotFoundError(f"No .hdf5 files under {self.data_dir}")
        joint_action_max = None
        joint_action_min = None
        for ep_idx, fname in enumerate(h5_files):
            fpath = os.path.join(self.data_dir, fname)
            camera_storage: Dict[str, str] = {}
            camera_names: Dict[str, Optional[str]] = {}
            with h5py.File(fpath, "r", libver="latest", swmr=self.use_swmr) as hf:
                T = self._infer_length(hf)
                # Resolve cameras and storage hints
                for cam in self.cameras:
                    actual = self._resolve_camera_name(hf, cam)
                    camera_names[cam] = actual
                    if actual is None:
                        if not self.allow_missing_cameras:
                            tried = list(self.camera_alias.get(cam, (cam,)))
                            raise KeyError(
                                f"Camera '{cam}' not found in episode {fname}. Tried aliases: {tried}"
                            )
                    else:
                        camera_storage[cam] = self._detect_camera_storage(hf, actual)

                # calculate joint action min/max
                cur_joint_vector = np.asarray(
                    hf["/joint_action/vector"], dtype=np.float32
                )
                local_max = cur_joint_vector.max(axis=0)  # shape (L,)
                local_min = cur_joint_vector.min(axis=0)
                if joint_action_max is None:
                    joint_action_max = local_max
                    joint_action_min = local_min
                else:
                    joint_action_max = np.maximum(joint_action_max, local_max)
                    joint_action_min = np.minimum(joint_action_min, local_min)

                # Optional numeric preload
                if self.preload_numeric:
                    numeric = {
                        "left_endpose": np.asarray(
                            hf["/endpose/left_endpose"], dtype=np.float32
                        ),
                        "right_endpose": np.asarray(
                            hf["/endpose/right_endpose"], dtype=np.float32
                        ),
                        "left_gripper": np.asarray(
                            hf["/endpose/left_gripper"], dtype=np.float32
                        ).reshape(-1),
                        "right_gripper": np.asarray(
                            hf["/endpose/right_gripper"], dtype=np.float32
                        ).reshape(-1),
                        "joint_action": np.asarray(
                            hf["/joint_action/vector"], dtype=np.float32
                        ),
                        "subtask_text": [s for s in hf["/subtask_text"]],
                    }
                    seen_blocks = []
                    history_list = []
                    prev = None

                    for s in numeric["subtask_text"]:
                        if s != prev:
                            if prev is not None and prev not in seen_blocks:
                                seen_blocks.append(prev)
                            prev = s
                        history_list.append(list(seen_blocks))
                    numeric["subtask_history"] = history_list
                    seen_blocks.append(numeric["subtask_text"][-1])
                    numeric["subtask_reasoning"] = [seen_blocks] * len(history_list)
                else:
                    raise ValueError("Now is only support numeric while need text")
                    numeric = None

                # Optional image preload
                images_ram: Optional[Dict[str, Any]] = None
                if self.preload_images:
                    if self.preload_images == "meta":
                        images_ram = {"__storage__": camera_storage}
                    else:
                        images_ram = {}
                        for cam in self.cameras:
                            actual = camera_names.get(cam)
                            if actual is None:
                                continue
                            dspath = self._rgb_dataset_path(actual)
                            ds = hf[dspath]
                            if camera_storage.get(cam) == "array":
                                images_ram[cam] = np.asarray(
                                    ds, dtype=np.uint8
                                )  # (T,H,W,3) or (T,3,H,W)
                            else:
                                images_ram[cam] = [
                                    bytes(
                                        b
                                        if isinstance(b, (bytes, bytearray))
                                        else getattr(b, "tobytes", lambda: bytes(b))()
                                    )
                                    for b in ds[:]
                                ]
            if not self.input_joint_norm:
                self.joint_action_max = torch.from_numpy(joint_action_max)
                self.joint_action_min = torch.from_numpy(joint_action_min)
            # Build episode meta and tables
            self._episodes.append(
                EpisodeMeta(
                    path=fpath,
                    T=T,
                    camera_names=camera_names,
                    camera_storage=camera_storage,
                )
            )
            self._ram_numeric.append(numeric)
            self._ram_images.append(images_ram)
            self._instr_cache.append(
                self._read_instructions(
                    os.path.join(
                        self.instructions_dir, os.path.splitext(fname)[0] + ".json"
                    )
                )
            )
            self._future_tables.append(
                self._build_future_table(T, self.chunk_size, self.frequency_ratio)
            )

            # Expand flat index
            if self.original_dataset_len:
                self._index.extend((ep_idx, t) for t in range(T))
            else:
                t_list = list(range(0, T, self.frequency_ratio))
                if t_list[-1] != T - 1:
                    t_list.append(T - 1)
                self._index.extend((ep_idx, t) for t in t_list)

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _natural_key(s: str):
        import re

        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)
        ]

    @staticmethod
    def _infer_length(hf: h5py.File) -> int:
        # Use left_gripper length as canonical T, fallback to other datasets
        candidates = [
            "/endpose/left_gripper",
            "/endpose/right_gripper",
            "/endpose/left_endpose",
            "/endpose/right_endpose",
        ]
        for k in candidates:
            if k in hf:
                return int(hf[k].shape[0])
        raise KeyError("Cannot infer episode length T from HDF5")

    def _detect_camera_storage(self, hf: h5py.File, camera_actual: str) -> str:
        dspath = self._rgb_dataset_path(camera_actual)
        if hf.get(dspath) is None:
            raise KeyError(f"Missing dataset {dspath}")
        ds = hf[dspath]
        # Heuristic: arrays will have ndim >= 4, encoded bytes are 1-D (T,)
        if ds.ndim >= 4:
            return "array"
        return "encoded"

    @staticmethod
    def _build_future_table(T: int, H: int, frequency_ratio: int = 1) -> np.ndarray:
        # For each t in [0..T-1], compute indices of t+1..t+H with clamp to T-1
        t = np.arange(T, dtype=np.int64)[:, None]
        offs = np.arange(1, H + 1, dtype=np.int64)[None, :] * frequency_ratio
        idx = t + offs
        idx[idx >= T] = T - 1
        return idx

    def _rgb_dataset_path(self, camera_actual: str) -> str:
        return f"/observation/{camera_actual}/rgb"

    def _load_rgb_frame(
        self,
        hf: h5py.File,
        camera_actual: str,
        t: int,
        storage_hint: Optional[str] = None,
    ) -> Image.Image:
        dspath = self._rgb_dataset_path(camera_actual)
        ds = hf[dspath]
        storage = storage_hint or ("array" if ds.ndim >= 4 else "encoded")
        if storage == "array":
            arr = ds[t]
            # return self._array_to_pil(arr)
            return self._array_to_cv2(arr)
        # encoded bytes path
        raw = ds[t]
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        elif hasattr(raw, "tobytes"):
            raw = raw.tobytes()
        else:
            raw = bytes(raw)
        # img = Image.open(io.BytesIO(raw))
        # img.load()
        nparr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img_rgb = img[:, :, ::-1]  # BGR -> RGB, match robotwin setting
        tensor_img = torch.from_numpy(np.array(img))
        return tensor_img

    @staticmethod
    def _array_to_pil(arr: np.ndarray) -> Image.Image:
        # Accept (H,W,3/4) or (3/4,H,W)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
        if arr.shape[-1] in (1, 3, 4):
            return Image.fromarray(arr.astype(np.uint8))
        if arr.shape[0] in (1, 3, 4):
            return Image.fromarray(np.moveaxis(arr, 0, -1).astype(np.uint8))
        # Fallback: best effort
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def _array_to_cv2(arr: np.ndarray) -> np.ndarray:
        """
        Accepts (H, W, 3/4) or (3/4, H, W) numpy arrays
        and returns an OpenCV-compatible image (numpy array, BGR order).
        """
        if arr.ndim != 3:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")

        # Case 1: (H, W, C) format
        if arr.shape[-1] in (1, 3, 4):
            img = arr.astype(np.uint8)

        # Case 2: (C, H, W) format → move channel axis
        elif arr.shape[0] in (1, 3, 4):
            img = np.moveaxis(arr, 0, -1).astype(np.uint8)

        else:
            # fallback
            img = arr.astype(np.uint8)
        # img_rgb = img[:, :, ::-1] #return RGB

        # if img.shape[-1] == 3:  # RGB → BGR
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # elif img.shape[-1] == 4:  # RGBA → BGRA
        #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)
        # return img

    @staticmethod
    def _read_vector(hf: h5py.File, path: str, t: int) -> np.ndarray:
        return np.asarray(hf[path][t], dtype=np.float32)

    @staticmethod
    def _read_vectors(hf: h5py.File, path: str, idxs: np.ndarray) -> np.ndarray:
        return np.asarray(hf[path][idxs], dtype=np.float32)

    @staticmethod
    def _read_scalar(hf: h5py.File, path: str, t: int) -> float:
        return float(np.asarray(hf[path][t]).reshape(()))

    @staticmethod
    def _read_scalars(hf: h5py.File, path: str, idxs: np.ndarray) -> np.ndarray:
        return np.asarray(hf[path][idxs], dtype=np.float32).reshape(-1)

    def _read_instructions(self, json_path: str) -> Dict[str, List[str]]:
        if not os.path.exists(json_path):
            return {"seen": [], "unseen": []}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            seen = [str(x) for x in data.get("seen", [])]
            unseen = [str(x) for x in data.get("unseen", [])]
            return {"seen": seen, "unseen": unseen}
        except Exception:
            # Do not fail training due to instruction parsing
            return {"seen": [], "unseen": []}

    def _resolve_camera_name(self, hf: h5py.File, logical_name: str) -> Optional[str]:
        """Resolve a logical camera name (e.g., 'head_camera') to an actual group
        name present in the HDF5 (e.g., 'front_camera'). Returns None if not found."""
        candidates = list(self.camera_alias.get(logical_name, (logical_name,)))
        for cand in candidates:
            if hf.get(self._rgb_dataset_path(cand)) is not None:
                return cand
        return None

    def concat_endpose(self, left_endpose, left_gripper, right_endpose, right_gripper):
        """
        Concatenate left/right endpose and gripper values.

        Supports two types of input:
        - Single frame:
                left_endpose, right_endpose: [7]
                left_gripper, right_gripper: []
            -> output shape: [16]

        - Sequence:
                left_endpose, right_endpose: [T, 7]
                left_gripper, right_gripper: [T]
            -> output shape: [T, 16]
        """

        # Case 1: single frame
        if left_endpose.dim() == 1:
            left_full = torch.cat([left_endpose, left_gripper.unsqueeze(0)])
            right_full = torch.cat([right_endpose, right_gripper.unsqueeze(0)])
            return torch.cat([left_full, right_full])

        # Case 2: sequence
        elif left_endpose.dim() == 2:
            left_full = torch.cat([left_endpose, left_gripper.unsqueeze(1)], dim=1)
            right_full = torch.cat([right_endpose, right_gripper.unsqueeze(1)], dim=1)
            return torch.cat([left_full, right_full], dim=1)

        else:
            raise ValueError("left_endpose must be shape [7] or [T, 7]")

    def process_endpose(self, endpose, mode="quat", order="xyz"):
        """
        Process one or a batch of endposes.

        Parameters
        ----------
        endpose : ndarray
            Shape [7] (single) or [B, 7] (batch).
            Format [x, y, z, qx, qy, qz, qw].
        mode : str
            "quat"  -> return [x, y, z, qx, qy, qz, qw]
            "euler" -> return [x, y, z, roll, pitch, yaw] (radians, scaled here /5.0)
        order : str
            Euler rotation order, default "xyz".
            Could also be "zyx" depending on your convention.

        Returns
        -------
        ndarray
            Same shape as input but last dimension = 7 (quat mode) or 6 (euler mode).
        """
        endpose = np.asarray(endpose)

        # Single case [7]
        if endpose.ndim == 1:
            xyz = endpose[:3]
            quat = endpose[3:]
            if mode == "quat":
                return np.concatenate([xyz, quat])
            elif mode == "euler":
                r = R.from_quat(quat)
                roll, pitch, yaw = r.as_euler(order, degrees=False)
                return np.concatenate([xyz, [roll / 5.0, pitch / 5.0, yaw / 5.0]])
            else:
                raise ValueError("mode must be 'quat' or 'euler'")

        # Batch case [B, 7]
        elif endpose.ndim == 2 and endpose.shape[1] == 7:
            xyz = endpose[:, :3]
            quat = endpose[:, 3:]
            if mode == "quat":
                return np.concatenate([xyz, quat], axis=1)
            elif mode == "euler":
                r = R.from_quat(quat)
                euler = r.as_euler(order, degrees=False) / 5.0  # scale like your code
                return np.concatenate([xyz, euler], axis=1)
            else:
                raise ValueError("mode must be 'quat' or 'euler'")
        else:
            raise ValueError("Input must be shape [7] or [B, 7]")

    # def compute_delta_pose(self, pose_now, pose_next):
    #     """
    #     Compute delta between two 8D poses: [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]

    #     Args:
    #         pose_now: torch.Tensor or np.ndarray, shape (8,)
    #         pose_next: torch.Tensor or np.ndarray, shape (8,)

    #     Returns:
    #         torch.Tensor, shape (8,) -> delta pose
    #     """
    #     # Convert to numpy if input is a torch tensor
    #     pose_now_np = pose_now.detach().cpu().numpy() if isinstance(pose_now, torch.Tensor) else pose_now
    #     pose_next_np = pose_next.detach().cpu().numpy() if isinstance(pose_next, torch.Tensor) else pose_next

    #     # Position delta
    #     delta_pos = pose_next_np[:3] - pose_now_np[:3]

    #     # Quaternion relative rotation, old,use scipy
    #     # q_now = R.from_quat(pose_now_np[3:7])   # [x, y, z, w]
    #     # q_next = R.from_quat(pose_next_np[3:7])
    #     # delta_q = q_next * q_now.inv()
    #     # delta_quat = delta_q.as_quat()

    #     # # Normalize to avoid numerical drift
    #     # delta_quat /= np.linalg.norm(delta_quat)

    #     delta_quat = t3d.quaternions.qmult(t3d.quaternions.qinverse(pose_now_np[3:7]), pose_next_np[3:7])
    #     delta_quat /= np.linalg.norm(delta_quat)
    #     # Concatenate and convert back to torch
    #     delta = np.concatenate([delta_pos, delta_quat, pose_next_np[7:8]]) #delta xyz+deltaq+gripper
    #     return torch.from_numpy(delta).to(pose_now.device).type_as(pose_now)

    # def make_action_chunk(self, pose_t, future_poses):
    #     """
    #     Build an action chunk sequence from current pose and K future poses.

    #     Args:
    #         pose_t: torch.Tensor, shape (14,) -> [left(7) + right(7)]
    #         future_poses: torch.Tensor, shape (K, 14) -> future poses

    #     Returns:
    #         torch.Tensor, shape (K, 14) -> action sequence
    #     """
    #     K = future_poses.shape[0]
    #     actions = []

    #     # First action: future[0] - pose_t
    #     delta_left = self.compute_delta_pose(pose_t[:8], future_poses[0, :8])
    #     delta_right = self.compute_delta_pose(pose_t[8:], future_poses[0, 8:])
    #     actions.append(torch.cat([delta_left, delta_right]))

    #     # Following actions: future[k] - future[k-1]
    #     for k in range(1, K):
    #         delta_left = self.compute_delta_pose(future_poses[k-1, :8], future_poses[k, :8])
    #         delta_right = self.compute_delta_pose(future_poses[k-1, 8:], future_poses[k, 8:])
    #         actions.append(torch.cat([delta_left, delta_right]))

    #     return torch.stack(actions, dim=0)

    # ----------------------------
    # Dataset API
    # ----------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, t = self._index[idx]
        meta = self._episodes[ep_idx]
        T = meta.T
        future_idx = self._future_tables[ep_idx][t]  # shape (H,)
        images: List[torch.Tensor] = []  # head left right
        if self.predict_image:
            predict_images: List[torch.Tensor] = []
        ram_images = self._ram_images[ep_idx]
        # Decide if we need to open HDF5 for any camera
        need_open = False
        for cam in self.cameras:
            actual = meta.camera_names.get(cam)
            if actual is None:
                continue
            if not (isinstance(ram_images, dict) and cam in ram_images):
                need_open = True
                break

        hf = None
        try:
            if need_open:
                hf = h5py.File(meta.path, "r", libver="latest", swmr=self.use_swmr)
            for cam in self.cameras:
                actual = meta.camera_names.get(cam)
                if actual is None:
                    # camera missing in this episode; skip
                    continue
                # Prefer RAM if available for this cam
                if isinstance(ram_images, dict) and cam in ram_images:
                    storage = meta.camera_storage.get(cam)
                    if storage == "array":
                        arr = ram_images[cam][t]
                        # img = self._array_to_pil(arr)
                        img = self._array_to_cv2(arr)

                        if self.predict_image:
                            pred_arr = ram_images[cam][future_idx[-1]]
                            pred_img = self._array_to_cv2(pred_arr)

                    else:
                        raw = ram_images[cam][t]
                        # img = Image.open(io.BytesIO(raw)); img.load()
                        nparr = np.frombuffer(raw, dtype=np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        # img = img[:, :, ::-1]
                        if self.predict_image:
                            pred_raw = ram_images[cam][future_idx[-1]]
                            pred_nparr = np.frombuffer(pred_raw, dtype=np.uint8)
                            pred_img = cv2.imdecode(pred_nparr, cv2.IMREAD_COLOR)
                else:
                    # Fallback: read-on-demand from HDF5
                    storage_hint = meta.camera_storage.get(cam)
                    img = self._load_rgb_frame(hf, actual, t, storage_hint=storage_hint)
                    if self.predict_image:
                        pred_img = self._load_rgb_frame(
                            hf, actual, future_idx[-1], storage_hint=storage_hint
                        )

                images.append(
                    self.image_transform(img, resolution=self.resolution)
                    if self.image_transform
                    else img
                )
                if self.predict_image:
                    predict_images.append(
                        self.image_transform(pred_img, resolution=self.resolution)
                        if self.image_transform
                        else pred_img
                    )

        finally:
            if hf is not None:
                hf.close()

        # --- Numerics ---
        ram_numeric = self._ram_numeric[ep_idx]
        to_t = lambda x: torch.as_tensor(x, dtype=self.float_dtype)

        if self.action_type == "ee":
            if ram_numeric is not None:
                left_endpose = ram_numeric["left_endpose"][t]
                right_endpose = ram_numeric["right_endpose"][t]
                left_gripper = ram_numeric["left_gripper"][t]
                right_gripper = ram_numeric["right_gripper"][t]

                fut_left_endpose = ram_numeric["left_endpose"][future_idx]
                fut_right_endpose = ram_numeric["right_endpose"][future_idx]
                fut_left_gripper = ram_numeric["left_gripper"][future_idx]
                fut_right_gripper = ram_numeric["right_gripper"][future_idx]
            else:
                with h5py.File(
                    meta.path, "r", libver="latest", swmr=self.use_swmr
                ) as hf2:
                    left_endpose = self._read_vector(hf2, "/endpose/left_endpose", t)
                    right_endpose = self._read_vector(hf2, "/endpose/right_endpose", t)
                    left_gripper = self._read_scalar(hf2, "/endpose/left_gripper", t)
                    right_gripper = self._read_scalar(hf2, "/endpose/right_gripper", t)

                    fut_left_endpose = self._read_vectors(
                        hf2, "/endpose/left_endpose", future_idx
                    )
                    fut_right_endpose = self._read_vectors(
                        hf2, "/endpose/right_endpose", future_idx
                    )
                    fut_left_gripper = self._read_scalars(
                        hf2, "/endpose/left_gripper", future_idx
                    )
                    fut_right_gripper = self._read_scalars(
                        hf2, "/endpose/right_gripper", future_idx
                    )

            if self.rotation_type == "euler":
                state_vals = self.concat_endpose(
                    to_t(self.process_endpose(to_t(left_endpose))),
                    to_t(left_gripper),
                    to_t(self.process_endpose(to_t(right_endpose))),
                    to_t(right_gripper),
                )

                action_vals = self.concat_endpose(
                    to_t(self.process_endpose(to_t(fut_left_endpose))),
                    to_t(fut_left_gripper),
                    to_t(self.process_endpose(to_t(fut_right_endpose))),
                    to_t(fut_right_gripper),
                )
                action_vals_raw = action_vals

            else:
                raise ValueError("Error rotation type")

        elif self.action_type == "joint":
            if ram_numeric is not None:
                cur_joint_action = ram_numeric["joint_action"][t]
                fut_joint_action = ram_numeric["joint_action"][future_idx]

            else:
                with h5py.File(
                    meta.path, "r", libver="latest", swmr=self.use_swmr
                ) as hf2:
                    cur_joint_action = self._read_scalar(hf2, "/joint_action/vector", t)
                    fut_joint_action = self._read_vectors(
                        hf2, "/joint_action/vector", future_idx
                    )

            def normalize(X, min_vals, max_vals, ignore_idx=(6, 13)):
                denom = max_vals - min_vals

                denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

                normed = 2 * (X - min_vals) / denom_safe - 1

                mask = denom == 0
                const_vals = min_vals
                normed = torch.where(mask, const_vals, normed)
                if ignore_idx:
                    idx = torch.tensor(ignore_idx, device=X.device, dtype=torch.long)
                    normed = normed.clone()
                    normed[..., idx] = X[..., idx]
                return normed

            state_vals_raw = to_t(cur_joint_action)  # state use eef+gripper
            action_vals_raw = to_t(fut_joint_action)
            action_vals = normalize(
                action_vals_raw, self.joint_action_min, self.joint_action_max
            )
            state_vals = normalize(
                state_vals_raw, self.joint_action_min, self.joint_action_max
            )
        else:
            raise ValueError("Error action type")
        try:
            instructions = self._instr_cache[ep_idx]["seen"][0]  #
        except:
            print("No valid instructions")
            instructions = None
        description = None
        if ram_numeric is not None and self.predict_text:
            subtask_text = ram_numeric["subtask_text"][t].decode("utf-8")
            subtask_history = [
                s.decode("utf-8") for s in ram_numeric["subtask_history"][t]
            ]
            subtask_reasoning = [
                s.decode("utf-8") for s in ram_numeric["subtask_reasoning"][t]
            ]

            reasoning_str = ", ".join(s.rstrip(".") for s in subtask_reasoning)
            history_str = ", ".join(s.rstrip(".") for s in subtask_history)

            if len(subtask_history) != 0:
                description = (
                    f"My task is {instructions}. "
                    f"I need to finish this task by {reasoning_str}. "
                    f"Currently, I have finished {history_str}. "
                    f"So now I should continue to {subtask_text}."
                )
            else:
                description = (
                    f"My task is {instructions}. "
                    f"I need to finish this task by {reasoning_str}. "
                    f"So now I should continue to {subtask_text}."
                )
        elif not self.predict_text:
            pass
        else:
            raise ValueError("text co training in None ram_numeric is Not ready now")
            with h5py.File(meta.path, "r", libver="latest", swmr=self.use_swmr) as hf2:
                subtask_text = hf2["subtask_text"][t].decode("utf-8")

        state_tokens = (
            torch.tensor(
                quantize_state(state_vals, bins=self.action_vocab_size),
                dtype=torch.long,
            )
            + self.vocab_offset
        )
        raw_action_tokens = quantize_action(action_vals, bins=self.action_vocab_size)
        flat_action_tokens = [a for chunk in raw_action_tokens for a in chunk]
        action_tokens = (
            torch.tensor(flat_action_tokens, dtype=torch.long) + self.vocab_offset
        )

        # dummy, match training script
        prev_action_tokens = torch.tensor([], dtype=torch.long)
        return {
            "text": instructions,
            "images": images,
            "predict_image": (
                predict_images[0] if self.predict_image else None
            ),  # only predict head image
            "state_tokens": state_tokens,
            "action_tokens": action_tokens,
            "prev_action_tokens": prev_action_tokens,
            "action_dim": self.action_dim,
            # "ral_action": action_vals_raw,
            # "action": action_vals
            # "subtask_text": subtask_text,
            # "subtask_history": subtask_history,
            # "subtask_reasoning": subtask_reasoning,
            "description": description if description else None,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        images = [b["images"] for b in batch]
        state_tokens = [b["state_tokens"] for b in batch]
        prev_action_tokens = [b["prev_action_tokens"] for b in batch]
        action_tokens = [b["action_tokens"] for b in batch]
        action_dims = [b["action_dim"] for b in batch]
        descriptions = [b["description"] for b in batch]
        predict_images = [b["predict_image"] for b in batch]
        return (
            images,
            texts,
            state_tokens,
            prev_action_tokens,
            action_tokens,
            action_dims,
            descriptions,
            predict_images,
        )
