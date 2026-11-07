"""Quart page, trame state, CSV lookup, three linked orthogonal slices."""

import io
import json
import os
import threading
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, quote

import numpy as np
import pandas as pd
from PIL import Image
from quart import Quart, Response, request
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOImage import vtkNIFTIImageReader

from trame.app import TrameApp
from trame.decorators import controller

CSV_PATH = Path(__file__).with_name("cases.csv")
MASKS = {"seg": "seg_path", "spleen": "spleen_path"}
PLANES = ("axial", "coronal", "sagittal")
# Radiologist: patient right on the viewer's left. Neuro: patient right on the viewer's right.
# Both keep anterior and the head up. Sagittal faces the patient's right (anterior on the left).
VIEWS = ("Radiologist", "Neuro")
# Screen-right and screen-down directions in RAS for each plane.
_SCREEN = {
    "Radiologist": {
        "axial": ((-1, 0, 0), (0, -1, 0)),
        "coronal": ((-1, 0, 0), (0, 0, -1)),
        "sagittal": ((0, -1, 0), (0, 0, -1)),
    },
    "Neuro": {
        "axial": ((1, 0, 0), (0, -1, 0)),
        "coronal": ((1, 0, 0), (0, 0, -1)),
        "sagittal": ((0, -1, 0), (0, 0, -1)),
    },
}
# Column and row voxel axes of an unflipped slice. 0=i, 1=j, 2=k.
_PLANE_AXES = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}
# (level, width) for the window dropdown. Lung matches the previous fixed window.
WINDOWS = {
    "Lung": (-600, 1500),
    "SoftTissue": (50, 350),
    "Brain": (40, 80),
    "Bone": (500, 2000),
    "Abdomen": (40, 400),
}
# Label 0 is clear. One foreground label uses the first color. More than one
# foreground label gets a different color per label id.
COLORMAP = (
    (0.90, 0.12, 0.12),
    (0.10, 0.78, 0.22),
    (0.16, 0.42, 0.95),
    (0.95, 0.78, 0.08),
    (0.88, 0.18, 0.78),
    (0.05, 0.82, 0.82),
    (0.96, 0.48, 0.08),
    (0.58, 0.22, 0.92),
    (0.52, 0.90, 0.18),
    (0.96, 0.45, 0.55),
    (0.18, 0.58, 0.42),
    (0.72, 0.28, 0.18),
    (0.28, 0.72, 0.95),
    (0.78, 0.90, 0.35),
    (0.95, 0.25, 0.40),
    (0.40, 0.28, 0.72),
)

app = Quart(__name__)
_volumes = {}
_luts = {}
_lock = threading.RLock()


def table():
    return pd.read_csv(CSV_PATH, dtype=str)


def case_ids():
    return table()["case"].tolist()


def other_mask(mask):
    names = tuple(MASKS)
    if mask == names[0] and len(names) > 1:
        return names[1]
    return names[0]


def lookup(case, mask):
    column = MASKS.get(mask or "seg")
    if column is None:
        raise LookupError(f"unknown mask: {mask}")
    hit = table().loc[lambda rows: rows["case"] == case]
    if hit.empty:
        raise LookupError(f"unknown id: {case}")
    image_path, mask_path = hit.iloc[0]["image_path"], hit.iloc[0][column]
    for path in (image_path, mask_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return image_path, mask_path


def volume(path):
    mtime = os.path.getmtime(path)
    hit = _volumes.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    reader = vtkNIFTIImageReader()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    if image.GetNumberOfScalarComponents() != 1:
        raise ValueError(f"{path} is not a scalar image")
    dims = image.GetDimensions()
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    array = np.ascontiguousarray(scalars.reshape(dims[2], dims[1], dims[0]))
    matrix = reader.GetSFormMatrix() or reader.GetQFormMatrix()
    if matrix is None:
        axes = np.eye(3)
    else:
        axes = np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
    _volumes[path] = (mtime, array, axes)
    return array


def flips_for(axes, view):
    flips = {}
    for plane, (col, row) in _PLANE_AXES.items():
        right, down = _SCREEN[view][plane]
        flips[plane] = [
            bool(np.dot(axes[:, col], right) < 0),
            bool(np.dot(axes[:, row], down) < 0),
        ]
    return flips


def orient(ct_slice, mask_slice, flip):
    if flip[0]:
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)
    if flip[1]:
        ct_slice = np.flipud(ct_slice)
        mask_slice = np.flipud(mask_slice)
    return ct_slice, mask_slice


def color_lut(mask):
    labels = np.unique(mask.astype(np.int32, copy=False))
    foreground = labels[labels > 0]
    top = int(foreground.max()) if foreground.size else 0
    lut = np.zeros((top + 1, 3), np.float32)
    if foreground.size == 1:
        lut[int(foreground[0])] = COLORMAP[0]
    elif foreground.size > 1:
        for label in foreground:
            lut[int(label)] = COLORMAP[(int(label) - 1) % len(COLORMAP)]
    return lut


def cached_lut(path, mask):
    mtime = os.path.getmtime(path)
    hit = _luts.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    lut = color_lut(mask)
    _luts[path] = (mtime, lut)
    return lut


def focus(mask):
    depth, height, width = mask.shape
    hits = np.flatnonzero(mask.reshape(depth, -1).any(axis=1))
    if len(hits) == 0:
        return width // 2, height // 2, depth // 2
    k = int(hits[len(hits) // 2])
    ys, xs = np.nonzero(mask[k] > 0)
    if len(xs) == 0:
        return width // 2, height // 2, k
    return int(xs.mean()), int(ys.mean()), k


def clamp(value, size):
    return min(max(int(value), 0), size - 1)


def crop_window(center, full, zoom):
    span = max(1, min(int(full), int(full / zoom)))
    start = int(center) - span // 2
    return max(0, min(start, int(full) - span)), span


def compose(ct_slice, mask_slice, overlay, lut, level, width, opacity, pngw):
    lo = level - width / 2
    hi = level + width / 2
    gray = np.clip((ct_slice.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if overlay and opacity > 0:
        labels = np.clip(mask_slice.astype(np.int32, copy=False), 0, lut.shape[0] - 1)
        strength = (labels > 0)[..., None].astype(np.float32) * opacity
        rgb = rgb * (1 - strength) + lut[labels] * strength
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    if image.width != pngw:
        pngh = max(1, int(round(pngw * image.height / image.width)))
        image = image.resize((pngw, pngh), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class Viewer(TrameApp):
    def __init__(self):
        self._ct = None
        self._mask = None
        self._axes = np.eye(3)
        self._lut = np.zeros((1, 3), np.float32)
        super().__init__()
        self.state.image_key = ""
        self.state.mask_key = "seg"
        self.state.image_path = ""
        self.state.mask_path = ""
        self.state.overlay = 1
        self.state.window = "Lung"
        self.state.level, self.state.width = WINDOWS["Lung"]
        self.state.opacity = 60
        self.state.view = "Radiologist"
        self.state.zoom = 1
        self.state.pngw = 512
        self.state.i = 0
        self.state.j = 0
        self.state.k = 0
        self.state.x = 1
        self.state.y = 1
        self.state.z = 1
        self.state.ready()

    @controller.set("apply")
    def apply(self, image_key, mask_key, overlay, i, j, k, window, opacity, view, zoom, pngw):
        mask_key = mask_key or "seg"
        if window not in WINDOWS:
            raise ValueError(f"unknown window: {window}")
        if view not in VIEWS:
            raise ValueError(f"unknown view: {view}")
        image_path, mask_path = lookup(image_key, mask_key)
        if image_path != self.state.image_path:
            _volumes.clear()
            _luts.clear()
        ct = volume(image_path)
        mask = volume(mask_path)
        if ct.shape != mask.shape:
            raise ValueError(f"shape mismatch {ct.shape} vs {mask.shape}")
        depth, height, width = ct.shape
        self._ct = ct
        self._mask = mask
        self._axes = _volumes[image_path][2]
        self._lut = cached_lut(mask_path, mask)
        if i is None or j is None or k is None:
            i, j, k = focus(mask)
        i, j, k = clamp(i, width), clamp(j, height), clamp(k, depth)
        if (image_path, mask_path) != (self.state.image_path, self.state.mask_path):
            print(f"{image_key} {image_path} | {mask_key} {mask_path}", flush=True)
        self.state.image_key = image_key
        self.state.mask_key = mask_key
        self.state.image_path = image_path
        self.state.mask_path = mask_path
        self.state.overlay = 1 if overlay else 0
        self.state.window = window
        self.state.level, self.state.width = WINDOWS[window]
        self.state.opacity = opacity
        self.state.view = view
        self.state.zoom = zoom
        self.state.pngw = pngw
        self.state.i = i
        self.state.j = j
        self.state.k = k
        self.state.x = width
        self.state.y = height
        self.state.z = depth
        self.state.flush()

    def png(self, plane):
        i, j, k = int(self.state.i), int(self.state.j), int(self.state.k)
        overlay = bool(int(self.state.overlay))
        zoom = float(self.state.zoom)
        i0, iw = crop_window(i, self.state.x, zoom)
        j0, jh = crop_window(j, self.state.y, zoom)
        k0, kd = crop_window(k, self.state.z, zoom)
        if plane == "axial":
            slices = (self._ct[k, j0:j0 + jh, i0:i0 + iw], self._mask[k, j0:j0 + jh, i0:i0 + iw])
        elif plane == "coronal":
            slices = (self._ct[k0:k0 + kd, j, i0:i0 + iw], self._mask[k0:k0 + kd, j, i0:i0 + iw])
        elif plane == "sagittal":
            slices = (self._ct[k0:k0 + kd, j0:j0 + jh, i], self._mask[k0:k0 + kd, j0:j0 + jh, i])
        else:
            raise ValueError("plane must be axial, coronal, or sagittal")
        slices = orient(*slices, flips_for(self._axes, self.state.view)[plane])
        level, width = int(self.state.level), int(self.state.width)
        opacity = int(self.state.opacity) / 100
        return compose(*slices, overlay, self._lut, level, width, opacity, int(self.state.pngw))


viewer = Viewer()


def one(found, name):
    values = found.get(name)
    return values[0] if values else ""


def parse_opacity(text):
    if text == "":
        return 60
    try:
        value = float(text)
    except ValueError:
        raise ValueError("opacity must be a number") from None
    if "." in text and abs(value) <= 1:
        value *= 100
    return min(100, max(0, int(round(value))))


def parse_zoom(text):
    if text == "":
        return 1.0
    try:
        value = float(text)
    except ValueError:
        raise ValueError("zoom must be a number") from None
    return round(min(32.0, max(1.0, value)), 2)


def parse_pngw(text):
    if text == "":
        return 512
    try:
        value = int(text)
    except ValueError:
        raise ValueError("png width must be an integer") from None
    return min(2048, max(64, value))


def query():
    raw = request.query_string.decode().replace(",", "&")
    found = parse_qs(raw, keep_blank_values=True)
    image, mask = one(found, "image"), one(found, "mask")
    overlay = one(found, "overlay") not in ("0", "false", "off")
    overlay2 = one(found, "overlay2") not in ("0", "false", "off")
    row2 = one(found, "row2") in ("1", "true", "on")
    mask2 = one(found, "mask2")
    window = one(found, "window") or "Lung"
    opacity = parse_opacity(one(found, "opacity"))
    view = one(found, "view") or "Radiologist"
    crosshair = one(found, "crosshair") not in ("0", "false", "off")
    zoom = parse_zoom(one(found, "zoom"))
    pngw = parse_pngw(one(found, "pngw"))
    shown = {plane: one(found, plane) not in ("0", "false", "off") for plane in PLANES}
    coords = []
    for name in ("i", "j", "k"):
        text = one(found, name)
        if text == "":
            coords.append(None)
            continue
        try:
            coords.append(int(text))
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
    return (
        image, mask, overlay, *coords, one(found, "plane"),
        row2, mask2, overlay2, window, opacity, view, crosshair, zoom, pngw, shown,
    )


def usage():
    links = " ".join(
        f'<a href="/?image={escape(case)}">{escape(case)}</a>' for case in case_ids()
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>trame-quart</title>
<p>{links}</p>
"""


def page(row2, mask2, overlay2, window, opacity, view, crosshair, zoom, pngw, shown):
    image = viewer.state.image_key
    mask = viewer.state.mask_key
    overlay = int(viewer.state.overlay)
    i, j, k = int(viewer.state.i), int(viewer.state.j), int(viewer.state.k)
    x, y, z = int(viewer.state.x), int(viewer.state.y), int(viewer.state.z)
    if mask2 not in MASKS:
        mask2 = other_mask(mask)
    flips = {name: flips_for(viewer._axes, name) for name in VIEWS}
    state = {
        "image": image,
        "mask": mask,
        "overlay": bool(overlay),
        "mask2": mask2,
        "overlay2": bool(overlay2),
        "row2": bool(row2),
        "window": window,
        "opacity": opacity,
        "view": view,
        "crosshair": bool(crosshair),
        "zoom": zoom,
        "pngw": pngw,
        "planes": shown,
        "flip": flips,
        "i": i,
        "j": j,
        "k": k,
        "x": x,
        "y": y,
        "z": z,
    }

    def options(selected):
        return "".join(
            f'<option value="{escape(name)}"{" selected" if name == selected else ""}>{escape(name)}</option>'
            for name in MASKS
        )

    def views(mask_name, show, include_src):
        parts = []
        for plane in PLANES:
            flip = flips[view][plane]
            key = (
                f"{mask_name}|{1 if show else 0}|{window}|{opacity}|{int(flip[0])}{int(flip[1])}"
                f"|{zoom:.2f}|{pngw}|{i}|{j}|{k}"
            )
            src_attr = ""
            if include_src and shown[plane]:
                src = (
                    f"/slice.png?image={quote(image)}&mask={quote(mask_name)}"
                    f"&overlay={1 if show else 0}&window={quote(window)}&opacity={opacity}"
                    f"&view={quote(view)}&zoom={zoom}&pngw={pngw}&plane={plane}&i={i}&j={j}&k={k}"
                )
                src_attr = f' src="{src}"'
            if plane == "sagittal":
                left, right = "A", "P"
            elif view == "Radiologist":
                left, right = "R", "L"
            else:
                left, right = "L", "R"
            fig_hidden = "" if shown[plane] else " hidden"
            parts.append(
                f"""<figure data-plane="{plane}"{fig_hidden}>
  <figcaption>{plane}</figcaption>
  <div class="frame" data-plane="{plane}">
    <img alt="{plane}" draggable="false" data-key="{key}"{src_attr}>
    <b class="tag l">{left}</b><b class="tag r">{right}</b>
    <i class="v"></i><i class="h"></i>
  </div>
</figure>"""
            )
        return "".join(parts)

    checked = " checked" if overlay else ""
    checked2 = " checked" if overlay2 else ""
    row_checked = " checked" if row2 else ""
    hidden = "" if row2 else " hidden"
    hair_checked = " checked" if crosshair else ""
    body = " ".join(part for part in (("two" if row2 else ""), ("" if crosshair else "nohair")) if part)
    window_options = "".join(
        f'<option value="{escape(name)}"{" selected" if name == window else ""}>{escape(name)}</option>'
        for name in WINDOWS
    )
    view_options = "".join(
        f'<option value="{escape(name)}"{" selected" if name == view else ""}>{escape(name)}</option>'
        for name in VIEWS
    )
    plane_checks = "".join(
        f'<label><input id="plane-{plane}" type="checkbox"{" checked" if shown[plane] else ""}> {plane}</label>'
        for plane in PLANES
    )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{escape(image)} / {escape(mask)}</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 14px sans-serif; }}
  header, .bar {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; padding: 8px 12px; }}
  input[type="range"] {{ width: 120px; accent-color: #3dffa6; }}
  input[type="number"] {{ width: 72px; background: #222; color: #eee; border: 1px solid #444; }}
  .views {{ display: flex; gap: 8px; align-items: flex-start; flex-wrap: wrap; padding: 0 12px 12px; }}
  figure {{ margin: 0; }}
  figcaption {{ margin-bottom: 4px; }}
  .frame {{ position: relative; display: inline-block; background: #000; cursor: crosshair; user-select: none; }}
  img {{ display: block; max-height: 78vh; max-width: min(32vw, 100%); image-rendering: pixelated; user-select: none; }}
  body.two img {{ max-height: 36vh; }}
  .tag {{ position: absolute; top: 4px; color: #3dffa6; font: 12px sans-serif; pointer-events: none; text-shadow: 0 0 3px #000; }}
  .tag.l {{ left: 4px; }}
  .tag.r {{ right: 4px; }}
  .v, .h {{ position: absolute; background: #3dffa6; pointer-events: none; }}
  .v {{ width: 1px; top: 0; bottom: 0; }}
  .h {{ height: 1px; left: 0; right: 0; }}
  body.nohair .v, body.nohair .h {{ display: none; }}
  body.nohair .frame {{ cursor: default; }}
  p {{ margin: 0; color: #aaa; }}
</style>
<body class="{body}">
<header>
  <label>view <select id="view">{view_options}</select></label>
  {plane_checks}
  <label>window <select id="window">{window_options}</select></label>
  <label>overlay <select id="mask">{options(mask)}</select></label>
  <label><input id="show" type="checkbox"{checked}> show</label>
  <label>opacity <input id="opacity" type="range" min="0" max="100" step="1" value="{opacity}"> <span id="opacity-val">{opacity}%</span></label>
  <label><input id="row2" type="checkbox"{row_checked}> second row</label>
  <label><input id="hair" type="checkbox"{hair_checked}> crosshair</label>
  <label>png width <input id="pngw" type="number" min="64" max="2048" step="64" value="{pngw}"></label>
  <p id="ijk">i {i} j {j} k {k} zoom {zoom:.2f}×</p>
</header>
<div class="views" id="row-1">{views(mask, bool(overlay), True)}</div>
<section id="row-2"{hidden}>
  <div class="bar">
    <label>overlay <select id="mask2">{options(mask2)}</select></label>
    <label><input id="show2" type="checkbox"{checked2}> show</label>
  </div>
  <div class="views">{views(mask2, bool(overlay2), row2)}</div>
</section>
<script>
const state = {json.dumps(state)};
const planes = {json.dumps(list(PLANES))};
function src(plane, mask, overlay) {{
  const q = new URLSearchParams({{
    image: state.image, mask, overlay: overlay ? "1" : "0",
    window: state.window, opacity: state.opacity, view: state.view,
    zoom: state.zoom, pngw: state.pngw,
    plane, i: state.i, j: state.j, k: state.k,
  }});
  return "/slice.png?" + q.toString();
}}
function crop(center, full, zoom) {{
  const span = Math.max(1, Math.min(full, Math.floor(full / zoom)));
  let start = center - Math.floor(span / 2);
  if (start < 0) start = 0;
  if (start > full - span) start = full - span;
  return [start, span];
}}
function frac(index, start, span, flipped) {{
  const pos = index - start + 0.5;
  return (flipped ? span - pos : pos) / span;
}}
function place() {{
  const f = state.flip[state.view];
  const iw = crop(state.i, state.x, state.zoom);
  const jh = crop(state.j, state.y, state.zoom);
  const kd = crop(state.k, state.z, state.zoom);
  const at = {{
    axial: [frac(state.i, iw[0], iw[1], f.axial[0]), frac(state.j, jh[0], jh[1], f.axial[1])],
    coronal: [frac(state.i, iw[0], iw[1], f.coronal[0]), frac(state.k, kd[0], kd[1], f.coronal[1])],
    sagittal: [frac(state.j, jh[0], jh[1], f.sagittal[0]), frac(state.k, kd[0], kd[1], f.sagittal[1])],
  }};
  const radio = state.view === "Radiologist";
  document.querySelectorAll(".frame").forEach((frame) => {{
    const plane = frame.dataset.plane;
    const xy = at[plane];
    frame.querySelector(".v").style.left = (xy[0] * 100) + "%";
    frame.querySelector(".h").style.top = (xy[1] * 100) + "%";
    const sagittal = plane === "sagittal";
    frame.querySelector(".tag.l").textContent = sagittal ? "A" : radio ? "R" : "L";
    frame.querySelector(".tag.r").textContent = sagittal ? "P" : radio ? "L" : "R";
  }});
  document.getElementById("ijk").textContent = "i " + state.i + "  j " + state.j + "  k " + state.k + "  zoom " + state.zoom.toFixed(2) + "×";
}}
function refresh(reload) {{
  const url = new URL(location.href);
  url.searchParams.set("image", state.image);
  url.searchParams.set("mask", state.mask);
  url.searchParams.set("overlay", state.overlay ? "1" : "0");
  url.searchParams.set("mask2", state.mask2);
  url.searchParams.set("overlay2", state.overlay2 ? "1" : "0");
  url.searchParams.set("row2", state.row2 ? "1" : "0");
  url.searchParams.set("window", state.window);
  url.searchParams.set("opacity", state.opacity);
  url.searchParams.set("view", state.view);
  url.searchParams.set("crosshair", state.crosshair ? "1" : "0");
  url.searchParams.set("zoom", state.zoom);
  url.searchParams.set("pngw", state.pngw);
  for (const plane of planes) url.searchParams.set(plane, state.planes[plane] ? "1" : "0");
  url.searchParams.set("i", state.i);
  url.searchParams.set("j", state.j);
  url.searchParams.set("k", state.k);
  history.replaceState(null, "", url);
  document.body.classList.toggle("two", state.row2);
  document.body.classList.toggle("nohair", !state.crosshair);
  document.getElementById("row-2").hidden = !state.row2;
  const rows = [
    {{id: "row-1", mask: state.mask, overlay: state.overlay, on: true}},
    {{id: "row-2", mask: state.mask2, overlay: state.overlay2, on: state.row2}},
  ];
  for (const row of rows) {{
    if (!row.on) continue;
    const root = document.getElementById(row.id);
    for (const plane of planes) {{
      const fig = root.querySelector('figure[data-plane="' + plane + '"]');
      fig.hidden = !state.planes[plane];
      if (fig.hidden) continue;
      const img = fig.querySelector("img");
      const flip = state.flip[state.view][plane];
      const key = row.mask + "|" + (row.overlay ? 1 : 0) + "|" + state.window + "|" + state.opacity + "|" + (flip[0] ? 1 : 0) + (flip[1] ? 1 : 0) + "|" + state.zoom.toFixed(2) + "|" + state.pngw + "|" + state.i + "|" + state.j + "|" + state.k;
      if (reload || img.dataset.key !== key || !img.getAttribute("src")) {{
        img.dataset.key = key;
        img.src = src(plane, row.mask, row.overlay);
      }}
    }}
  }}
  place();
}}
let drag = null;
function voxel(ev) {{
  const img = drag.img;
  if (!img.naturalWidth) return null;
  const rect = img.getBoundingClientRect();
  let col = Math.min(img.naturalWidth - 1, Math.max(0, Math.floor((ev.clientX - rect.left) / rect.width * img.naturalWidth)));
  let row = Math.min(img.naturalHeight - 1, Math.max(0, Math.floor((ev.clientY - rect.top) / rect.height * img.naturalHeight)));
  const flip = state.flip[state.view][drag.plane];
  const basis = drag.mode === "cross" ? drag : state;
  const along = {{
    axial: [crop(basis.i, state.x, basis.zoom), crop(basis.j, state.y, basis.zoom)],
    coronal: [crop(basis.i, state.x, basis.zoom), crop(basis.k, state.z, basis.zoom)],
    sagittal: [crop(basis.j, state.y, basis.zoom), crop(basis.k, state.z, basis.zoom)],
  }}[drag.plane];
  function at(pixel, length, axis, flipped) {{
    if (flipped) pixel = length - 1 - pixel;
    const pos = Math.min(axis[1] - 1, Math.floor(pixel / length * axis[1]));
    return axis[0] + pos;
  }}
  return [at(col, img.naturalWidth, along[0], flip[0]), at(row, img.naturalHeight, along[1], flip[1])];
}}
function aim(ev) {{
  const hit = voxel(ev);
  if (!hit) return;
  if (drag.plane === "axial") {{ state.i = hit[0]; state.j = hit[1]; }}
  if (drag.plane === "coronal") {{ state.i = hit[0]; state.k = hit[1]; }}
  if (drag.plane === "sagittal") {{ state.j = hit[0]; state.k = hit[1]; }}
  refresh(false);
}}
function zoomTo(dy) {{
  state.zoom = Math.round(Math.max(1, Math.min(32, drag.zoom * Math.exp(dy / 160))) * 100) / 100;
  state.i = drag.i;
  state.j = drag.j;
  state.k = drag.k;
  refresh(false);
}}
document.querySelectorAll(".frame").forEach((frame) => {{
  frame.addEventListener("pointerdown", (ev) => {{
    if (ev.button !== 1) return;
    const img = frame.querySelector("img");
    if (!img.naturalWidth) return;
    drag = {{ mode: "zoom", plane: frame.dataset.plane, img, y: ev.clientY, zoom: state.zoom }};
    aim(ev);
    drag.i = state.i;
    drag.j = state.j;
    drag.k = state.k;
    frame.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  }});
  frame.addEventListener("mousedown", (ev) => {{
    if (ev.button === 1) {{ ev.preventDefault(); return; }}
    if (ev.button !== 0) return;
    const img = frame.querySelector("img");
    if (!img.naturalWidth) return;
    drag = {{ mode: "cross", plane: frame.dataset.plane, img, i: state.i, j: state.j, k: state.k, zoom: state.zoom }};
    aim(ev);
    ev.preventDefault();
  }});
  frame.addEventListener("auxclick", (ev) => {{
    if (ev.button === 1) ev.preventDefault();
  }});
  let wheel = 0;
  frame.addEventListener("wheel", (ev) => {{
    ev.preventDefault();
    let pixels = ev.deltaY;
    if (ev.deltaMode === 1) pixels *= 16;
    if (ev.deltaMode === 2) pixels *= frame.clientHeight || 400;
    wheel += pixels;
    const steps = Math.trunc(wheel / 48);
    if (steps === 0) return;
    wheel -= steps * 48;
    const plane = frame.dataset.plane;
    if (plane === "axial") state.k = Math.max(0, Math.min(state.z - 1, state.k + steps));
    if (plane === "coronal") state.j = Math.max(0, Math.min(state.y - 1, state.j + steps));
    if (plane === "sagittal") state.i = Math.max(0, Math.min(state.x - 1, state.i + steps));
    refresh(false);
  }}, {{ passive: false }});
}});
window.addEventListener("pointermove", (ev) => {{
  if (!drag || drag.mode !== "zoom") return;
  zoomTo(drag.y - ev.clientY);
}});
window.addEventListener("mousemove", (ev) => {{
  if (!drag) return;
  if (drag.mode === "zoom") return;
  if ((ev.buttons & 1) === 0) {{ drag = null; return; }}
  aim(ev);
}});
window.addEventListener("pointerup", (ev) => {{
  if (drag && drag.mode === "zoom" && ev.button === 1) drag = null;
}});
window.addEventListener("mouseup", (ev) => {{
  if (!drag || drag.mode === "zoom") return;
  drag = null;
}});
document.getElementById("mask").onchange = () => {{
  state.mask = document.getElementById("mask").value;
  refresh(true);
}};
document.getElementById("show").onchange = () => {{
  state.overlay = document.getElementById("show").checked;
  refresh(true);
}};
document.getElementById("row2").onchange = () => {{
  state.row2 = document.getElementById("row2").checked;
  refresh(true);
}};
document.getElementById("mask2").onchange = () => {{
  state.mask2 = document.getElementById("mask2").value;
  refresh(true);
}};
document.getElementById("show2").onchange = () => {{
  state.overlay2 = document.getElementById("show2").checked;
  refresh(true);
}};
document.getElementById("window").onchange = () => {{
  state.window = document.getElementById("window").value;
  refresh(true);
}};
document.getElementById("view").onchange = () => {{
  state.view = document.getElementById("view").value;
  refresh(true);
}};
for (const plane of planes) {{
  document.getElementById("plane-" + plane).onchange = () => {{
    state.planes[plane] = document.getElementById("plane-" + plane).checked;
    refresh(false);
  }};
}}
document.getElementById("hair").onchange = () => {{
  state.crosshair = document.getElementById("hair").checked;
  refresh(false);
}};
document.getElementById("pngw").onchange = () => {{
  const value = Math.round(Number(document.getElementById("pngw").value));
  state.pngw = Math.max(64, Math.min(2048, value || 512));
  document.getElementById("pngw").value = state.pngw;
  refresh(true);
}};
let opacityTimer = 0;
document.getElementById("opacity").oninput = () => {{
  state.opacity = Number(document.getElementById("opacity").value);
  document.getElementById("opacity-val").textContent = state.opacity + "%";
  clearTimeout(opacityTimer);
  opacityTimer = setTimeout(() => refresh(true), 60);
}};
place();
</script>
</body>
"""


def load(need_plane=False):
    image, mask, overlay, i, j, k, plane, row2, mask2, overlay2, window, opacity, view, crosshair, zoom, pngw, shown = query()
    if not image:
        return None
    with _lock:
        viewer.ctrl.apply(image, mask, overlay, i, j, k, window, opacity, view, zoom, pngw)
        if need_plane:
            if plane not in PLANES:
                raise ValueError("plane must be axial, coronal, or sagittal")
            return viewer.png(plane)
        return page(row2, mask2, overlay2, window, opacity, view, crosshair, zoom, pngw, shown)


def fail(exc):
    status = 400 if isinstance(exc, ValueError) else 404
    return str(exc), status


@app.get("/")
async def index():
    try:
        body = load()
    except (LookupError, FileNotFoundError, ValueError) as exc:
        return fail(exc)
    if body is None:
        return usage()
    return body


@app.get("/slice.png")
async def slice_png():
    try:
        body = load(need_plane=True)
    except (LookupError, FileNotFoundError, ValueError) as exc:
        return fail(exc)
    if body is None:
        return "image is required", 400
    return Response(body, content_type="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
