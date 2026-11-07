"""Quart page, trame state, CSV lookup, NIfTI slice overlay."""

import csv
import io
import json
import os
import threading
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
from PIL import Image
from quart import Quart, Response, request
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOImage import vtkNIFTIImageReader

from trame.app import TrameApp
from trame.decorators import change, controller

CSV_PATH = Path(__file__).with_name("cases.csv")
# Lung window, Hounsfield units.
LEVEL = -600
WIDTH = 1500

app = Quart(__name__)
_volumes = {}
_lock = threading.Lock()


def lookup(key):
    with CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["key"] == key:
                path = row["path"]
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                return path
    raise LookupError(f"unknown id: {key}")


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
    _volumes[path] = (mtime, array)
    return array


def compose(ct_slice, mask_slice):
    lo = LEVEL - WIDTH / 2
    hi = LEVEL + WIDTH / 2
    gray = np.clip((ct_slice.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    hit = mask_slice > 0
    rgb[hit] = rgb[hit] * 0.4 + np.array([1.0, 0.15, 0.15]) * 0.6
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class Viewer(TrameApp):
    def __init__(self):
        self._ct = None
        self._mask = None
        self.png = b""
        super().__init__()
        self.state.slice_index = -1
        self.state.slice_max = 0
        self.state.image_key = ""
        self.state.mask_key = ""
        self.state.image_path = ""
        self.state.mask_path = ""
        self.state.ready()

    @controller.set("apply")
    def apply(self, image_key, mask_key, slice_index):
        image_path = lookup(image_key)
        mask_path = lookup(mask_key)
        ct = volume(image_path)
        mask = volume(mask_path)
        if ct.shape != mask.shape:
            raise ValueError(f"shape mismatch {ct.shape} vs {mask.shape}")
        depth = ct.shape[0]
        self._ct = ct
        self._mask = mask
        if (image_path, mask_path) != (self.state.image_path, self.state.mask_path):
            print(f"{image_key} {image_path} | {mask_key} {mask_path}", flush=True)
        self.state.image_key = image_key
        self.state.mask_key = mask_key
        self.state.image_path = image_path
        self.state.mask_path = mask_path
        self.state.slice_max = depth - 1
        if slice_index is None:
            hits = np.flatnonzero(mask.reshape(depth, -1).any(axis=1))
            slice_index = int(hits[len(hits) // 2]) if len(hits) else depth // 2
        slice_index = min(max(int(slice_index), 0), depth - 1)
        same = int(self.state.slice_index) == slice_index
        self.state.slice_index = slice_index
        # trame notifies @change only on flush
        self.state.flush()
        if same:
            self._draw(slice_index)
        return self.png

    @change("slice_index")
    def on_slice(self, slice_index, **_):
        if self._ct is None or int(slice_index) < 0:
            return
        self._draw(int(slice_index))

    def _draw(self, index):
        self.png = compose(self._ct[index], self._mask[index])


viewer = Viewer()


def query():
    raw = request.query_string.decode().replace(",", "&")
    found = parse_qs(raw, keep_blank_values=True)

    def one(name):
        values = found.get(name)
        return values[0] if values else ""

    image, mask, slice_text = one("image"), one("mask"), one("slice")
    if slice_text == "":
        return image, mask, None
    try:
        return image, mask, int(slice_text)
    except ValueError:
        raise ValueError("slice must be an integer") from None


def usage():
    href = "/?image=case0&mask=case0_lung"
    return f"""<!doctype html>
<meta charset="utf-8">
<title>trame-quart</title>
<p>Open <a href="{href}">{escape(href)}</a></p>
<p>Comma form: <a href="/?image=case0,mask=case0_lung">/?image=case0,mask=case0_lung</a></p>
"""


def page():
    image = viewer.state.image_key
    mask = viewer.state.mask_key
    index = int(viewer.state.slice_index)
    last = int(viewer.state.slice_max)
    src = f"/overlay.png?image={image}&mask={mask}&slice={index}"
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{escape(image)} / {escape(mask)}</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 14px sans-serif; }}
  main {{ padding: 12px; }}
  img {{ width: min(90vw, 640px); image-rendering: pixelated; background: #000; }}
  input {{ width: min(90vw, 640px); }}
  p {{ margin: 6px 0; word-break: break-all; }}
</style>
<main>
  <p>{escape(image)} → {escape(viewer.state.image_path)}</p>
  <p>{escape(mask)} → {escape(viewer.state.mask_path)}</p>
  <p>slice <output id="n">{index}</output> / {last}</p>
  <input id="s" type="range" min="0" max="{last}" value="{index}">
  <div><img id="v" alt="overlay" src="{escape(src)}"></div>
</main>
<script>
const image = {json.dumps(image)};
const mask = {json.dumps(mask)};
const slider = document.getElementById("s");
const view = document.getElementById("v");
const label = document.getElementById("n");
slider.oninput = () => {{
  label.textContent = slider.value;
  const url = new URL(location.href);
  url.searchParams.set("image", image);
  url.searchParams.set("mask", mask);
  url.searchParams.set("slice", slider.value);
  history.replaceState(null, "", url);
  view.src = "/overlay.png?image=" + encodeURIComponent(image)
    + "&mask=" + encodeURIComponent(mask)
    + "&slice=" + slider.value;
}};
</script>
"""


def load():
    image, mask, slice_index = query()
    if not image or not mask:
        return None
    with _lock:
        viewer.ctrl.apply(image, mask, slice_index)
        return page(), viewer.png


@app.get("/")
async def index():
    try:
        loaded = load()
    except (LookupError, FileNotFoundError, ValueError) as exc:
        status = 400 if isinstance(exc, ValueError) else 404
        return str(exc), status
    if loaded is None:
        return usage()
    return loaded[0]


@app.get("/overlay.png")
async def overlay():
    try:
        loaded = load()
    except (LookupError, FileNotFoundError, ValueError) as exc:
        status = 400 if isinstance(exc, ValueError) else 404
        return str(exc), status
    if loaded is None:
        return "image and mask are required", 400
    return Response(loaded[1], content_type="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
