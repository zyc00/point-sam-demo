import dataclasses
import os

import numpy as np
import torch
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from omegaconf import OmegaConf
from safetensors.torch import load_model
from scipy.spatial.transform import Rotation

from point_sam import build_point_sam
import argparse

app = Flask(__name__, static_folder="static")
CORS(app)


@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor
    features: torch.Tensor
    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int):
    if repeats == 1:
        return x
    shape = list(x.shape)
    shape.insert(dim + 1, 1)
    shape[dim + 1] = repeats
    x = x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
    return x


class PointCloudProcessor:
    def __init__(self, device="cuda", batch=True, return_tensors="pt"):
        self.device = device
        self.batch = batch
        self.return_tensors = return_tensors

        self.center = None
        self.scale = None

    def __call__(self, xyz: np.ndarray, rgb: np.ndarray):
        # # The original data is z-up. Make it y-up.
        # rot = Rotation.from_euler("x", -90, degrees=True)
        # xyz = rot.apply(xyz)

        if self.center is None or self.scale is None:
            self.center = xyz.mean(0)
            self.scale = np.max(np.linalg.norm(xyz - self.center, axis=-1))

        xyz = (xyz - self.center) / self.scale
        rgb = ((rgb / 255.0) - 0.5) * 2

        if self.return_tensors == "np":
            coords = np.float32(xyz)
            feats = np.float32(rgb)
            if self.batch:
                coords = np.expand_dims(coords, 0)
                feats = np.expand_dims(feats, 0)
        elif self.return_tensors == "pt":
            coords = torch.tensor(xyz, dtype=torch.float32, device=self.device)
            feats = torch.tensor(rgb, dtype=torch.float32, device=self.device)
            if self.batch:
                coords = coords.unsqueeze(0)
                feats = feats.unsqueeze(0)
        else:
            raise ValueError(self.return_tensors)

        return coords, feats

    def normalize(self, xyz):
        return (xyz - self.center) / self.scale


class PointCloudSAMPredictor:
    input_xyz: np.ndarray
    input_rgb: np.ndarray
    prompt_coords: list[tuple[float, float, float]]
    prompt_labels: list[int]

    coords: torch.Tensor
    feats: torch.Tensor

    pc_embedding: torch.Tensor
    patches: dict[str, torch.Tensor]
    prompt_mask: torch.Tensor

    def __init__(self):
        print("Created model")
        model = build_point_sam("./model-2.safetensors")
        model.pc_encoder.patch_embed.grouper.num_groups = 1024
        model.pc_encoder.patch_embed.grouper.group_size = 128
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        self.model = model

        self.input_rgb = None
        self.input_xyz = None

        self.input_processor = None
        self.coords = None
        self.feats = None

        self.pc_embedding = None
        self.patches = None

        self.prompt_coords = None
        self.prompt_labels = None
        self.prompt_mask = None
        self.candidate_index = 0

    @torch.no_grad()
    def set_pointcloud(self, xyz, rgb):
        self.input_xyz = xyz
        self.input_rgb = rgb

        self.input_processor = PointCloudProcessor()
        coords, feats = self.input_processor(xyz, rgb)
        self.coords = coords
        self.feats = feats

        pc_embedding, patches = self.model.pc_encoder(self.coords, self.feats)
        self.pc_embedding = pc_embedding
        self.patches = patches
        self.prompt_mask = None

    def set_prompts(self, prompt_coords, prompt_labels):
        self.prompt_coords = prompt_coords
        self.prompt_labels = prompt_labels

    @torch.no_grad()
    def predict_mask(self):
        normalized_prompt_coords = self.input_processor.normalize(
            np.array(self.prompt_coords)
        )
        prompt_coords = torch.tensor(
            normalized_prompt_coords, dtype=torch.float32, device="cuda"
        )
        prompt_labels = torch.tensor(
            self.prompt_labels, dtype=torch.bool, device="cuda"
        )
        prompt_coords = prompt_coords.reshape(1, -1, 3)
        prompt_labels = prompt_labels.reshape(1, -1)

        multimask_output = prompt_coords.shape[1] == 1

        # [B * M, num_outputs, num_points], [B * M, num_outputs]
        def decode_masks(
            coords,
            feats,
            pc_embedding,
            patches,
            prompt_coords,
            prompt_labels,
            prompt_masks,
            multimask_output,
        ):
            pc_embeddings, patches = pc_embedding, patches
            centers = patches["centers"]
            knn_idx = patches["knn_idx"]
            coords = patches["coords"]
            feats = patches["feats"]
            aux_inputs = AuxInputs(coords=coords, features=feats, centers=centers)

            pc_pe = self.model.point_encoder.pe_layer(centers)
            sparse_embeddings = self.model.point_encoder(prompt_coords, prompt_labels)
            dense_embeddings = self.model.mask_encoder(
                prompt_masks, coords, centers, knn_idx
            )
            dense_embeddings = repeat_interleave(
                dense_embeddings,
                sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                0,
            )

            logits, iou_preds = self.model.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=multimask_output,
            )
            return logits, iou_preds

        logits, scores = decode_masks(
            self.coords,
            self.feats,
            self.pc_embedding,
            self.patches,
            prompt_coords,
            prompt_labels,
            (
                self.prompt_mask[self.candidate_index].unsqueeze(0)
                if self.prompt_mask is not None
                else None
            ),
            multimask_output,
        )
        logits = logits.squeeze(0)
        scores = scores.squeeze(0)

        # if multimask_output:
        #     index = scores.argmax(0).item()
        #     logit = logits[index]
        # else:
        #     logit = logits.squeeze(0)

        # self.prompt_mask = logit.unsqueeze(0)

        # pred_mask = logit > 0
        # return pred_mask.cpu().numpy()

        # Sort according to scores
        _, indices = scores.sort(descending=True)
        logits = logits[indices]

        self.prompt_mask = logits  # [num_outputs, num_points]
        self.candidate_index = 0

        return (logits > 0).cpu().numpy()

    def set_candidate(self, index):
        self.candidate_index = index


predictor = PointCloudSAMPredictor()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/assets/<path:path>")
def assets_route(path):
    print(path)
    return app.send_static_file(f"assets/{path}")


@app.route("/hello_world", methods=["GET"])
def hello_world():
    return "Hello, World!"


@app.route("/set_pointcloud", methods=["POST"])
def set_pointcloud():
    request_data = request.get_json()
    # print(request_data)
    # print(type(request_data["points"]))
    # print(type(request_data["colors"]))

    xyz = request_data["points"]
    xyz = np.array(xyz).reshape(-1, 3)
    rgb = request_data["colors"]
    rgb = np.array(list(rgb)).reshape(-1, 3)
    predictor.set_pointcloud(xyz, rgb)

    pc_embedding = predictor.pc_embedding.cpu().numpy()
    patches = {
        "centers": predictor.patches["centers"].cpu().numpy().tolist(),
        "knn_idx": predictor.patches["knn_idx"].cpu().numpy().tolist(),
        "coords": predictor.coords.cpu().numpy().tolist(),
        "feats": predictor.feats.cpu().numpy().tolist(),
    }
    center = predictor.input_processor.center
    scale = predictor.input_processor.scale
    return jsonify(
        {
            "pc_embedding": pc_embedding.tolist(),
            "patches": patches,
            "center": center.tolist(),
            "scale": scale,
        }
    )


@app.route("/set_candidate", methods=["POST"])
def set_candidate():
    request_data = request.get_json()
    candidate_index = request_data["index"]
    predictor.set_candidate(candidate_index)
    return "success"


def visualize_pcd_with_prompts(xyz, rgb, prompt_coords, prompt_labels):
    import trimesh

    pcd = trimesh.PointCloud(xyz, rgb)
    prompt_spheres = []
    for i, coord in enumerate(prompt_coords):
        sphere = trimesh.creation.icosphere()
        sphere.apply_scale(0.02)
        sphere.apply_translation(coord)
        sphere.visual.vertex_colors = [255, 0, 0] if prompt_labels[i] else [0, 255, 0]
        prompt_spheres.append(sphere)

    return trimesh.Scene([pcd] + prompt_spheres)


@app.route("/set_prompts", methods=["POST"])
def set_prompts():
    request_data = request.get_json()
    print(request_data.keys())

    # [n_prompts, 3]
    prompt_coords = request_data["prompt_coords"]
    # [n_prompts]. 0 for negative, 1 for positive
    prompt_labels = request_data["prompt_labels"]
    embedding = torch.tensor(request_data["embeddings"]).cuda()
    patches = request_data["patches"]
    patches = {k: torch.tensor(v).cuda() for k, v in patches.items()}
    predictor.pc_embedding = embedding
    predictor.patches = patches
    predictor.input_processor.center = np.array(request_data["center"])
    predictor.input_processor.scale = request_data["scale"]
    try:
        if request_data["prompt_mask"] is not None:
            predictor.prompt_mask = torch.tensor(request_data["prompt_mask"]).cuda()
        else:
            predictor.prompt_mask = None
    except:
        predictor.prompt_mask = None
    # instance_id = request_data["instance_id"]  # int
    if len(prompt_coords) == 0:
        predictor.prompt_mask = None
        pred_mask = np.zeros([len(prompt_coords)], dtype=np.bool_)
        return jsonify({"mask": pred_mask.tolist()})

    predictor.set_prompts(prompt_coords, prompt_labels)
    pred_mask = predictor.predict_mask()
    prompt_mask = predictor.prompt_mask.cpu().numpy()

    # # Visualize
    # xyz = predictor.coords.cpu().numpy()[0]
    # rgb = predictor.feats.cpu().numpy()[0] * 0.5 + 0.5
    # prompt_coords = predictor.input_processor.normalize(np.array(predictor.prompt_coords))
    # scene = visualize_pcd_with_prompts(xyz, rgb, prompt_coords, predictor.prompt_labels)
    # scene.show()

    return jsonify({"mask": pred_mask.tolist(), "prompt_mask": prompt_mask.tolist()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
