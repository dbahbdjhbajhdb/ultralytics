# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construct aligned original/occluded image pairs without changing detection labels.

Occlusion grades choose source objects; synthetic rectangle coverage is recorded
separately and is never substituted for the real dataset's 0/1/2 grades.
"""

from __future__ import annotations

import math

import torch


def _number(config, name, default, lower=0.0, upper=1.0):
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Pair setting {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Pair setting {name} must lie in [{lower}, {upper}]")
    return value


def validate_pair_metadata(batch, nc=None):
    """Validate and return normalized boxes, image indices, classes, grades and validity.

    All returned tensors share the image device. Unknown grades must be marked
    ``occ_valid=False`` and remain ordinary detection targets.
    """
    for name in ("img", "bboxes", "batch_idx", "cls", "occlusion", "occ_valid"):
        if not isinstance(batch.get(name), torch.Tensor):
            raise ValueError(f"Paired occlusion training requires tensor metadata: {name}")
    image = batch["img"]
    if image.ndim != 4 or min(image.shape) < 1 or not image.is_floating_point():
        raise ValueError("Pair images must be nonempty floating-point BCHW tensors")
    boxes, indices, classes = batch["bboxes"], batch["batch_idx"], batch["cls"]
    grades, valid = batch["occlusion"], batch["occ_valid"]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Pair boxes must have shape (number of GTs, 4) in normalized xywh")
    count = len(boxes)
    if indices.ndim != 1 or indices.shape != (count,) or classes.shape != (count, 1):
        raise ValueError("Pair boxes, image indices and classes must align one-to-one")
    if grades.ndim != 1 or valid.ndim != 1 or len(grades) != count or len(valid) != count:
        raise ValueError("Pair boxes, occlusion degrees and validity masks must align one-to-one")
    if valid.dtype != torch.bool:
        raise ValueError("Pair occlusion validity masks must have boolean dtype")
    for name, value in (("boxes", boxes), ("image indices", indices), ("classes", classes), ("grades", grades)):
        if value.is_complex() or value.dtype == torch.bool or not torch.isfinite(value).all():
            raise ValueError(f"Pair {name} must contain finite real numeric values")
    device = image.device
    boxes = boxes.to(device=device, dtype=torch.float32)
    indices, classes = indices.to(device=device), classes.to(device=device).flatten()
    grades, valid = grades.to(device=device), valid.to(device=device)
    if ((boxes < -1e-6) | (boxes > 1 + 1e-6)).any() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Pair boxes must be normalized with strictly positive width and height")
    corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
    if ((corners < -1e-6) | (corners > 1 + 1e-6)).any():
        raise ValueError("Pair boxes must lie inside the transformed image")
    if (indices != indices.floor()).any() or ((indices < 0) | (indices >= image.shape[0])).any():
        raise ValueError("Pair image indices must be integers within the image batch")
    if (classes != classes.floor()).any() or (classes < 0).any() or (nc is not None and (classes >= nc).any()):
        raise ValueError("Pair classes must be nonnegative integer indices within the model's classes")
    if (grades != grades.floor()).any() or ((grades < -1) | (grades > 2)).any():
        raise ValueError("Pair grades must be integer 0/1/2, or -1 for unknown targets")
    if (valid & ((grades < 0) | (grades > 2))).any():
        raise ValueError("Known pair occlusion targets must use levels 0, 1 or 2")
    return boxes, indices.long(), classes.long(), grades.long(), valid


def _pixel_boxes(boxes, image):
    scale = boxes.new_tensor((image.shape[-1], image.shape[-2]))
    center, size = boxes[:, :2] * scale, boxes[:, 2:] * scale
    return torch.cat((center - size / 2, center + size / 2), dim=1)


@torch.no_grad()
def select_reliable_targets(decoded, batch, indices, config):
    """Gate GTs with a correct-class, confident, well-localized EMA prediction.

    ``decoded`` must be standard YOLO ``(B, 4 + nc, anchors)`` predictions with
    pixel xywh boxes and class probabilities, before NMS. This gate supplies no
    teacher boxes/classes as detection targets: the original GTs are retained.
    """
    if not isinstance(decoded, torch.Tensor) or decoded.ndim != 3 or decoded.shape[1] <= 4:
        raise ValueError("EMA reliability requires decoded YOLO predictions with shape (B, 4 + nc, anchors)")
    if not decoded.is_floating_point() or decoded.shape[0] != batch["img"].shape[0]:
        raise ValueError("EMA prediction batch and image batch must align")
    nc = decoded.shape[1] - 4
    boxes, image_indices, classes, _, _ = validate_pair_metadata(batch, nc=nc)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise ValueError("Reliability target indices must be a one-dimensional tensor")
    if (
        indices.is_complex()
        or indices.dtype == torch.bool
        or not torch.isfinite(indices).all()
        or (indices != indices.floor()).any()
        or ((indices < 0) | (indices >= len(boxes))).any()
    ):
        raise ValueError("Reliability target indices must refer to existing GT rows")
    indices = indices.to(device=boxes.device, dtype=torch.long)
    result = torch.zeros(len(indices), device=boxes.device, dtype=torch.bool)
    threshold_conf = _number(config, "teacher_conf", 0.15)
    threshold_iou = _number(config, "teacher_iou", 0.5)
    chunk_size = config.get("teacher_gt_chunk", 32)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("teacher_gt_chunk must be a positive integer")
    gt_boxes = _pixel_boxes(boxes, batch["img"])
    for image_index in range(batch["img"].shape[0]):
        positions = torch.where(image_indices[indices] == image_index)[0]
        if not len(positions):
            continue
        prediction = decoded[image_index].detach().float().to(device=boxes.device).transpose(0, 1)
        scores = prediction[:, 4:]
        confidence, predicted_class = scores.max(dim=1)
        usable = torch.isfinite(prediction).all(dim=1) & (confidence >= threshold_conf)
        usable &= (prediction[:, 2:4] > 0).all(dim=1)
        prediction, predicted_class = prediction[usable], predicted_class[usable]
        if not len(prediction):
            continue
        predicted_boxes = torch.cat(
            (prediction[:, :2] - prediction[:, 2:4] / 2, prediction[:, :2] + prediction[:, 2:4] / 2), dim=1
        )
        predicted_area = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(dim=1)
        for start in range(0, len(positions), chunk_size):
            selected_positions = positions[start : start + chunk_size]
            selected_rows = indices[selected_positions]
            regions = gt_boxes[selected_rows]
            top_left = torch.maximum(regions[:, None, :2], predicted_boxes[None, :, :2])
            bottom_right = torch.minimum(regions[:, None, 2:], predicted_boxes[None, :, 2:])
            intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
            gt_area = (regions[:, 2:] - regions[:, :2]).prod(dim=1)
            iou = intersection / (gt_area[:, None] + predicted_area[None] - intersection).clamp_min(1e-6)
            correct_class = classes[selected_rows, None] == predicted_class[None]
            result[selected_positions] = ((iou >= threshold_iou) & correct_class).any(dim=1)
    return result


def _rectangle(region, coverage, style, placement):
    """Return an integer rectangle wholly inside a GT, or None for tiny regions."""
    x1, y1, x2, y2 = region
    left, top, right, bottom = math.ceil(x1), math.ceil(y1), math.floor(x2), math.floor(y2)
    width, height = right - left, bottom - top
    if width < 4 or height < 4:
        return None
    desired_area = (x2 - x1) * (y2 - y1) * coverage
    if style == 0:
        rect_width, rect_height = max(4, round(desired_area / height)), height
    elif style == 1:
        rect_width, rect_height = width, max(4, round(desired_area / width))
    else:
        rect_width = max(4, round(math.sqrt(desired_area * width / height)))
        rect_height = max(4, round(desired_area / rect_width))
    if rect_width > width or rect_height > height:
        return None
    # Left/right/center placements cover a bounded part of the existing object;
    # no pasted object introduces an unlabelled detection target.
    offsets = (0.0, 1.0, 0.5)
    ratio = offsets[placement]
    rect_left = left + round((width - rect_width) * ratio)
    rect_top = top + round((height - rect_height) * ratio)
    return (rect_left, rect_top, rect_left + rect_width, rect_top + rect_height)


@torch.no_grad()
def make_occlusion_pairs(batch, config, progress, teacher_predictions=None):
    """Return a batch copy with at most one safe synthetic occluder per image.

    Only ``img`` changes. ``pair_indices`` contains global GT rows and
    ``pair_rectangles`` contains integer-valued pixel xyxy rectangles in exactly
    that order. ``statistics`` records why a batch might have no eligible pairs.
    Providing EMA predictions restricts augmentation to reliable original GTs.
    """
    boxes, image_indices, _, grades, valid = validate_pair_metadata(batch)
    if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress):
        raise ValueError("Pair schedule progress must be finite and numeric")
    if not 0 <= progress <= 1:
        raise ValueError("Pair schedule progress must lie in [0, 1]")
    selection = config.get("selection", "grade")
    if selection not in {"grade", "random"}:
        raise ValueError("Pair selection must be grade or random")
    probability = _number(config, "pair_probability", 0.5) * progress
    coverage_min = _number(config, "coverage_min", 0.15)
    coverage_max = _number(config, "coverage_max", 0.35)
    neighbor_limit = _number(config, "neighbor_max_coverage", 0.35)
    if not 0 < coverage_min <= coverage_max < 1 or neighbor_limit >= 1:
        raise ValueError("Pair coverage must be positive and partial, and neighbor coverage must be below one")
    attempts = config.get("pair_attempts", 8)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 8:
        raise ValueError("pair_attempts must be an integer between one and eight")
    source_mask = valid & ((grades == 0) if selection == "grade" else ((grades == 0) | (grades == 1)))
    candidates = torch.where(source_mask)[0]
    statistics = {
        "eligible_gt": len(candidates),
        "reliable_gt": len(candidates) if teacher_predictions is None else 0,
        "paired_images": 0,
        "paired_gt": 0,
        "paired_by_occlusion": [0, 0, 0],
        "rejected_geometry": 0,
        "rejected_neighbors": 0,
        "coverage_sum": 0.0,
        "effective_pair_probability": probability,
        "selection": selection,
    }
    result = batch.copy()
    image, pixel_boxes = batch["img"], _pixel_boxes(boxes, batch["img"])
    pair_indices, rectangles = [], []
    changed_image = None
    # A zero schedule is an exact no-op and does not advance augmentation RNG.
    if probability > 0 and len(candidates):
        if teacher_predictions is not None:
            reliable = select_reliable_targets(teacher_predictions, batch, candidates, config)
            candidates = candidates[reliable]
            statistics["reliable_gt"] = len(candidates)
        for image_index in range(image.shape[0]):
            selected = candidates[image_indices[candidates] == image_index]
            if not len(selected) or float(torch.rand((), device=image.device)) >= probability:
                continue
            rows_in_image = torch.where(image_indices == image_index)[0]
            for _ in range(attempts):
                candidate = int(selected[torch.randint(len(selected), (), device=image.device)])
                region = pixel_boxes[candidate].tolist()
                desired_coverage = coverage_min + (coverage_max - coverage_min) * float(
                    torch.rand((), device=image.device)
                )
                style = int(torch.randint(3, (), device=image.device))
                placement = int(torch.randint(3, (), device=image.device))
                rectangle = _rectangle(region, desired_coverage, style, placement)
                if rectangle is None:
                    statistics["rejected_geometry"] += 1
                    continue
                left, top, right, bottom = rectangle
                target_area = (region[2] - region[0]) * (region[3] - region[1])
                actual_coverage = (right - left) * (bottom - top) / target_area
                if not coverage_min - 1e-6 <= actual_coverage <= coverage_max + 1e-6:
                    statistics["rejected_geometry"] += 1
                    continue
                neighbors = rows_in_image[rows_in_image != candidate]
                if len(neighbors):
                    neighbor_boxes = pixel_boxes[neighbors]
                    rectangle_tensor = boxes.new_tensor(rectangle)
                    overlap = (
                        torch.minimum(neighbor_boxes[:, 2:], rectangle_tensor[2:])
                        - torch.maximum(neighbor_boxes[:, :2], rectangle_tensor[:2])
                    ).clamp_min(0).prod(dim=1)
                    neighbor_area = (neighbor_boxes[:, 2:] - neighbor_boxes[:, :2]).prod(dim=1).clamp_min(1e-6)
                    limits = torch.full_like(neighbor_area, neighbor_limit)
                    severe = valid[neighbors] & (grades[neighbors] == 2)
                    # Severe real GTs must not receive any additional erasure,
                    # including as a neighbor of the selected grade-0 target.
                    limits[severe] = 0.0
                    if (overlap / neighbor_area > limits + 1e-6).any():
                        statistics["rejected_neighbors"] += 1
                        continue
                if changed_image is None:
                    changed_image = image.clone()
                fill = image[image_index].mean(dim=(-2, -1), keepdim=True)
                changed_image[image_index, :, top:bottom, left:right] = fill
                pair_indices.append(candidate)
                rectangles.append(rectangle)
                statistics["paired_images"] += 1
                statistics["paired_gt"] += 1
                statistics["paired_by_occlusion"][int(grades[candidate])] += 1
                statistics["coverage_sum"] += actual_coverage
                break
    result["img"] = image if changed_image is None else changed_image
    result["pair_indices"] = torch.tensor(pair_indices, device=image.device, dtype=torch.long)
    result["pair_rectangles"] = torch.tensor(rectangles, device=image.device, dtype=torch.float32).reshape(-1, 4)
    result["statistics"] = statistics
    return result
