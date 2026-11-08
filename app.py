"""Trame NIfTI viewer: CSV lookup, three linked orthogonal VtkLocalViews."""

import asyncio
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import numpy as np
import pandas as pd
import vtk
from trame.app import get_server
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import client, html, vtk as vtk_widgets, vuetify3
from trame_vtk.modules.vtk.serializers.registry import (
    register_instance_serializer,
    register_js_class,
)
from trame_vtk.modules.vtk.serializers.serialize import serialize
from trame_vtk.modules.vtk.serializers.utils import reference_id, wrap_id
from vtk.util.numpy_support import vtk_to_numpy

CSV_PATH = Path(__file__).with_name("cases.csv")
MASKS = {"seg": "seg_path", "spleen": "spleen_path"}
PLANES = ("axial", "coronal", "sagittal")
VIEWS = ("Radiologist", "Neuro")
# Screen-right and screen-down in RAS. Radiologist puts patient right on the
# viewer's left. Both conventions keep anterior and the head up.
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
# In-plane column (screen-right axis) and row (screen-down axis). 0=x, 1=y, 2=z.
_PLANE_AXES = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}
_SLICE_MODE = {"axial": 2, "coronal": 1, "sagittal": 0}  # vtk.js K, J, I
_NORMAL = {"axial": 2, "coronal": 1, "sagittal": 0}
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
_HAIR = (0x3D / 255, 0xFF / 255, 0xA6 / 255)
_LUTS = {}

server = get_server(client_type="vue3")
state = server.state


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


def parse_index(found, name, alias):
    text = one(found, name)
    if text == "":
        text = one(found, alias)
    if text == "":
        return -1
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


def parse_search(search):
    raw = (search or "").lstrip("?").replace(",", "&")
    found = parse_qs(raw, keep_blank_values=True)
    case_id = one(found, "image")
    mask = one(found, "mask") or "seg"
    if mask not in MASKS:
        raise ValueError(f"unknown mask: {mask}")
    mask2 = one(found, "mask2") or other_mask(mask)
    if mask2 not in MASKS:
        raise ValueError(f"unknown mask: {mask2}")
    window_name = one(found, "window") or "Lung"
    if window_name not in WINDOWS:
        raise ValueError(f"unknown window: {window_name}")
    view_name = one(found, "view") or "Radiologist"
    if view_name not in VIEWS:
        raise ValueError(f"unknown view: {view_name}")
    return {
        "case_id": case_id,
        "mask": mask,
        "overlay": one(found, "overlay") not in ("0", "false", "off"),
        "x": parse_index(found, "x", "i"),
        "y": parse_index(found, "y", "j"),
        "z": parse_index(found, "z", "k"),
        "row2": one(found, "row2") in ("1", "true", "on"),
        "mask2": mask2,
        "overlay2": one(found, "overlay2") not in ("0", "false", "off"),
        "window_name": window_name,
        "opacity": parse_opacity(one(found, "opacity")),
        "view_name": view_name,
        "crosshair": one(found, "crosshair") not in ("0", "false", "off"),
        "zoom": parse_zoom(one(found, "zoom")),
        "show_axial": one(found, "axial") not in ("0", "false", "off"),
        "show_coronal": one(found, "coronal") not in ("0", "false", "off"),
        "show_sagittal": one(found, "sagittal") not in ("0", "false", "off"),
    }


def direction_matrix(image):
    matrix = image.GetDirectionMatrix()
    return np.array(
        [[matrix.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float
    )


def sform_axes(reader):
    matrix = reader.GetSFormMatrix() or reader.GetQFormMatrix()
    if matrix is None:
        return np.eye(3)
    axes = np.array(
        [[matrix.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float
    )
    if abs(np.linalg.det(axes)) < 1e-8:
        return np.eye(3)
    return axes


def screen_basis(axes, plane, view):
    """Index-space look direction and view-up. Dominant axis only, no oblique MPR."""
    right_ras, down_ras = _SCREEN[view][plane]
    col, row = _PLANE_AXES[plane]
    right = np.zeros(3)
    up = np.zeros(3)
    right[col] = 1.0 if np.dot(axes[:, col], right_ras) >= 0 else -1.0
    down_sign = 1.0 if np.dot(axes[:, row], down_ras) >= 0 else -1.0
    up[row] = -down_sign
    dop = np.cross(up, right)
    dop /= np.linalg.norm(dop)
    return dop, up


def world_vector(image, index_vector):
    return direction_matrix(image) @ np.array(index_vector, float)


def world_point(image, x, y, z):
    origin = np.array(image.GetOrigin(), float)
    spacing = np.array(image.GetSpacing(), float)
    return origin + direction_matrix(image) @ (np.array([x, y, z], float) * spacing)


def index_of(image, world):
    origin = np.array(image.GetOrigin(), float)
    spacing = np.array(image.GetSpacing(), float)
    local = np.linalg.solve(direction_matrix(image), np.array(world, float) - origin)
    return np.round(local / spacing).astype(int)


def display_world(focal, dop, up, scale, aspect, px, py, width, height):
    dop = np.array(dop, float)
    up = np.array(up, float)
    dop /= np.linalg.norm(dop)
    up /= np.linalg.norm(up)
    right = np.cross(dop, up)
    right /= np.linalg.norm(right)
    nx = float(px) / float(width) - 0.5
    ny = float(py) / float(height) - 0.5
    return (
        np.array(focal, float)
        + right * (nx * 2 * scale * aspect)
        + up * (ny * 2 * scale)
    )


def axis_length(image, axis):
    spacing = image.GetSpacing()
    dims = image.GetDimensions()
    return float(np.linalg.norm(direction_matrix(image)[:, axis]) * spacing[axis] * dims[axis])


def parallel_scale(image, plane, zoom, aspect):
    col, row = _PLANE_AXES[plane]
    width = axis_length(image, col)
    height = axis_length(image, row)
    aspect = aspect if aspect and aspect > 0 else 1.0
    return max(height, width / aspect) / 2.0 / max(float(zoom), 1.0)


def clamp(value, size):
    return min(max(int(value), 0), int(size) - 1)


def focus_of(image):
    dims = image.GetDimensions()
    img_w, img_d, img_h = dims
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    mask = np.ascontiguousarray(scalars.reshape(img_h, img_d, img_w))
    hits = np.flatnonzero(mask.reshape(img_h, -1).any(axis=1))
    if len(hits) == 0:
        return img_w // 2, img_d // 2, img_h // 2
    z = int(hits[len(hits) // 2])
    ys, xs = np.nonzero(mask[z] > 0)
    if len(xs) == 0:
        return img_w // 2, img_d // 2, z
    return int(xs.mean()), int(ys.mean()), z


def build_colors(scalars):
    labels = np.unique(scalars.astype(np.int32, copy=False))
    foreground = labels[labels > 0]
    top = int(foreground.max()) if foreground.size else 0
    count = top + 1
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(count)
    lut.SetTableRange(0, max(top, 1))
    lut.SetRange(0, max(top, 1))
    for label in range(count):
        lut.SetTableValue(label, 0, 0, 0, 0)
    if foreground.size == 1:
        chosen = {int(foreground[0]): COLORMAP[0]}
    else:
        chosen = {
            int(label): COLORMAP[(int(label) - 1) % len(COLORMAP)] for label in foreground
        }
    for label, rgb in chosen.items():
        lut.SetTableValue(label, float(rgb[0]), float(rgb[1]), float(rgb[2]), 1)
    lut.Build()
    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0, 0, 0, 0, 0.5, 1)
    for label, rgb in chosen.items():
        ctf.AddRGBPoint(float(label), float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.5, 1)
    pwf = vtk.vtkPiecewiseFunction()
    pwf.AddPoint(0.0, 0.0)
    pwf.AddPoint(0.5, 0.0)
    if top >= 1:
        pwf.AddPoint(0.5001, 1.0)
        pwf.AddPoint(float(top), 1.0)
    return lut, ctf, pwf


def colors_for(path, image):
    mtime = os.path.getmtime(path)
    hit = _LUTS.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    built = build_colors(scalars)
    _LUTS[path] = (mtime, built)
    return built


def read_volume(reader, path):
    mtime = os.path.getmtime(path)
    if getattr(reader, "loaded_path", None) == path and getattr(reader, "loaded_mtime", None) == mtime:
        return reader.GetOutput()
    reader.SetFileName(path)
    reader.Update()
    image = reader.GetOutput()
    if image is None or image.GetPointData().GetScalars() is None:
        raise ValueError(f"empty image: {path}")
    if image.GetNumberOfScalarComponents() != 1:
        raise ValueError(f"{path} is not a scalar image")
    reader.loaded_path = path
    reader.loaded_mtime = mtime
    reader.loaded_axes = sform_axes(reader)
    return image


def raw_output(mapper):
    algo = mapper.GetInputAlgorithm() if mapper is not None else None
    seen = set()
    while algo is not None and id(algo) not in seen:
        seen.add(id(algo))
        name = algo.GetClassName()
        if "MapToColors" in name or "MapToWindowLevel" in name:
            algo = algo.GetInputAlgorithm()
            continue
        image = algo.GetOutput()
        if image is None or image.GetDimensions() == (0, 0, 0):
            return None
        if image.GetPointData() is None or image.GetPointData().GetScalars() is None:
            return None
        return image
    return None


def cache_properties(obj_id, context, props):
    from trame_vtk.modules.vtk.serializers.cache import cache_properties as cache

    return cache(obj_id, context, props)


def image_slice_serializer(parent, instance, obj_id, context, depth):
    dependencies = []
    calls = []
    mapper = instance.GetMapper()
    prop = instance.GetProperty()
    if mapper is not None:
        mapper_id = reference_id(mapper)
        mapped = serialize(instance, mapper, mapper_id, context, depth + 1)
        if mapped:
            dependencies.append(mapped)
            calls.append(["setMapper", [wrap_id(mapper_id)]])
    if prop is not None:
        prop_id = reference_id(prop)
        mapped = serialize(instance, prop, prop_id, context, depth + 1)
        if mapped:
            dependencies.append(mapped)
            calls.append(["setProperty", [wrap_id(prop_id)]])
    if not dependencies:
        return None
    return {
        "parent": reference_id(parent),
        "id": obj_id,
        "type": "vtkImageSlice",
        "properties": cache_properties(
            obj_id, context, {"visibility": int(instance.GetVisibility())}
        ),
        "calls": calls,
        "dependencies": dependencies,
    }


def reslice_serializer(parent, mapper, obj_id, context, depth):
    image = raw_output(mapper)
    if image is None:
        return None
    image_id = reference_id(image)
    image_state = serialize(mapper, image, image_id, context, depth + 1)
    if not image_state:
        return None
    return {
        "parent": reference_id(parent),
        "id": obj_id,
        "type": "vtkImageMapper",
        "properties": cache_properties(
            obj_id,
            context,
            {
                "slicingMode": int(getattr(mapper, "slice_mode", 2)),
                "slice": int(getattr(mapper, "slice_index", 0)),
            },
        ),
        "calls": [["setInputData", [wrap_id(image_id)]]],
        "dependencies": [image_state],
    }


def image_property_serializer(parent, prop, obj_id, context, depth):
    dependencies = []
    calls = []
    for attr, method in (
        ("trame_ctf", "setRGBTransferFunction"),
        ("trame_pwf", "setScalarOpacity"),
    ):
        child = getattr(prop, attr, None)
        if child is None:
            continue
        child_id = reference_id(child)
        mapped = serialize(prop, child, child_id, context, depth + 1)
        if mapped:
            dependencies.append(mapped)
            calls.append([method, [wrap_id(child_id)]])
    use_range = 0
    if hasattr(prop, "GetUseLookupTableScalarRange"):
        use_range = 1 if prop.GetUseLookupTableScalarRange() else 0
    return {
        "parent": reference_id(parent),
        "id": obj_id,
        "type": "vtkImageProperty",
        "properties": cache_properties(
            obj_id,
            context,
            {
                "colorWindow": float(prop.GetColorWindow()),
                "colorLevel": float(prop.GetColorLevel()),
                "opacity": float(prop.GetOpacity()),
                "interpolationType": int(prop.GetInterpolationType()),
                "ambient": 1,
                "diffuse": 0,
                "useLookupTableScalarRange": use_range,
            },
        ),
        "calls": calls,
        "dependencies": dependencies,
    }


def camera_with_parallel(parent, instance, obj_id, context, _depth):
    return {
        "parent": reference_id(parent),
        "id": obj_id,
        "type": "vtkCamera",
        "properties": cache_properties(
            obj_id,
            context,
            {
                "focalPoint": instance.GetFocalPoint(),
                "position": instance.GetPosition(),
                "viewUp": instance.GetViewUp(),
                "clippingRange": instance.GetClippingRange(),
                "parallelProjection": 1 if instance.GetParallelProjection() else 0,
                "parallelScale": float(instance.GetParallelScale()),
            },
        ),
    }


def install_serializers(**_kwargs):
    from trame_vtk.modules.vtk.serializers.initialize import initialize_serializers

    initialize_serializers()
    for name in ("vtkImageSlice",):
        register_instance_serializer(name, image_slice_serializer)
    for name in ("vtkImageResliceMapper", "vtkOpenGLImageResliceMapper"):
        register_instance_serializer(name, reslice_serializer)
    register_instance_serializer("vtkImageProperty", image_property_serializer)
    for name in ("vtkCamera", "vtkOpenGLCamera"):
        register_instance_serializer(name, camera_with_parallel)
    register_js_class("vtkOpenGLCamera", "vtkCamera")
    # The client scene builder only constructs vtkRenderWindow.
    for name in (
        "vtkEGLRenderWindow",
        "vtkOSOpenGLRenderWindow",
        "vtkXOpenGLRenderWindow",
        "vtkOpenGLRenderWindow",
        "vtkGenericOpenGLRenderWindow",
        "vtkCocoaRenderWindow",
        "vtkWin32OpenGLRenderWindow",
        "vtkOffscreenOpenGLRenderWindow",
    ):
        register_js_class(name, "vtkRenderWindow")


def make_window():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(512, 512)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    renderer.ResetCameraClippingRange()
    return renderer, window, camera


def make_line():
    source = vtk.vtkLineSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*_HAIR)
    actor.GetProperty().SetLineWidth(1)
    actor.PickableOff()
    return source, actor


class Pane:
    def __init__(self, plane, row):
        self.plane = plane
        self.row = row
        self.ref = f"{plane}{row}"
        self.renderer, self.window, self.camera = make_window()
        self.slice_plane = vtk.vtkPlane()
        self.image_colors = vtk.vtkImageMapToWindowLevelColors()
        self.image_mapper = vtk.vtkImageResliceMapper()
        self.image_mapper.SetSlicePlane(self.slice_plane)
        self.image_mapper.SetInputConnection(self.image_colors.GetOutputPort())
        self.image_slice = vtk.vtkImageSlice()
        self.image_slice.SetMapper(self.image_mapper)
        self.image_slice.GetProperty().SetInterpolationTypeToNearest()
        self.mask_colors = vtk.vtkImageMapToColors()
        self.mask_mapper = vtk.vtkImageResliceMapper()
        self.mask_mapper.SetSlicePlane(self.slice_plane)
        self.mask_mapper.SetInputConnection(self.mask_colors.GetOutputPort())
        self.mask_slice = vtk.vtkImageSlice()
        self.mask_slice.SetMapper(self.mask_mapper)
        self.mask_slice.GetProperty().SetInterpolationTypeToNearest()
        # Sample stack. vtk.js has no ImageStack, so the slices are the view props.
        self.stack = vtk.vtkImageStack()
        self.stack.AddImage(self.image_slice)
        self.stack.AddImage(self.mask_slice)
        self.lines = [make_line(), make_line()]
        self.renderer.AddViewProp(self.image_slice)
        self.renderer.AddViewProp(self.mask_slice)
        for _source, actor in self.lines:
            self.renderer.AddViewProp(actor)
        self.view = None


class Viewer:
    def __init__(self):
        self.ct_reader = vtk.vtkNIFTIImageReader()
        self.mask_readers = {0: vtk.vtkNIFTIImageReader(), 1: vtk.vtkNIFTIImageReader()}
        self.panes = [Pane(plane, row) for row in (0, 1) for plane in PLANES]
        self._aspect = {}
        self._focal = None
        self._drag = None
        self._refreshing = False
        self._push_gen = 0
        self._ct_path = ""
        self._mask_paths = {0: "", 1: ""}
        state.case_id = ""
        state.mask = "seg"
        state.mask2 = "spleen"
        state.overlay = True
        state.overlay2 = True
        state.row2 = False
        state.window_name = "Lung"
        state.window_items = list(WINDOWS)
        state.mask_items = list(MASKS)
        state.view_items = list(VIEWS)
        state.view_name = "Radiologist"
        state.opacity = 60
        state.crosshair = True
        state.zoom = 1.0
        state.x = -1
        state.y = -1
        state.z = -1
        state.img_w = 1
        state.img_d = 1
        state.img_h = 1
        state.show_axial = True
        state.show_coronal = True
        state.show_sagittal = True
        state.has_case = False
        state.error = ""
        state.address = "/"
        state.ready()
        server.controller.on_server_ready.add(install_serializers)
        self._build_ui()
        install_serializers()

    def pane(self, ref):
        for pane in self.panes:
            if pane.ref == ref:
                return pane
        raise KeyError(ref)

    def _build_ui(self):
        # resize is emitted by the view, not the vtk.js interactor.
        events = (
            "['LeftButtonPress','LeftButtonRelease','MiddleButtonPress',"
            "'MiddleButtonRelease','MouseMove','MouseWheel']",
        )
        # One-element argument list. Trame splats it into open_query(search).
        mounted = (
            "[((window.__tq||(window.__tq=1,"
            "['pointerdown','mousedown','auxclick'].forEach(function(t){"
            "window.addEventListener(t,function(ev){if(ev.button===1)ev.preventDefault()},true)}"
            "))),window.location.search)]"
        )
        with VAppLayout(
            server, vuetify_config={"theme": {"defaultTheme": "dark"}}
        ):
            client.ClientTriggers(mounted=(self.open_query, mounted))
            client.ClientStateChange(
                value="address",
                change="$event ? window.history.replaceState(null, '', $event) : 0",
            )
            with html.Div(
                style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:8px 12px"
            ):
                vuetify3.VSelect(
                    v_model="view_name",
                    items=("view_items",),
                    label="View",
                    density="compact",
                    hide_details=True,
                    style="max-width:170px",
                )
                vuetify3.VCheckbox(
                    v_model="show_axial", label="axial", density="compact", hide_details=True
                )
                vuetify3.VCheckbox(
                    v_model="show_coronal",
                    label="coronal",
                    density="compact",
                    hide_details=True,
                )
                vuetify3.VCheckbox(
                    v_model="show_sagittal",
                    label="sagittal",
                    density="compact",
                    hide_details=True,
                )
                vuetify3.VSelect(
                    v_model="window_name",
                    items=("window_items",),
                    label="Window",
                    density="compact",
                    hide_details=True,
                    style="max-width:170px",
                )
                vuetify3.VSelect(
                    v_model="mask",
                    items=("mask_items",),
                    label="Overlay",
                    density="compact",
                    hide_details=True,
                    style="max-width:140px",
                )
                vuetify3.VCheckbox(
                    v_model="overlay", label="show", density="compact", hide_details=True
                )
                vuetify3.VSlider(
                    v_model="opacity",
                    min=0,
                    max=100,
                    step=1,
                    label="Opacity",
                    density="compact",
                    hide_details=True,
                    style="max-width:220px",
                )
                vuetify3.VCheckbox(
                    v_model="row2", label="second row", density="compact", hide_details=True
                )
                vuetify3.VCheckbox(
                    v_model="crosshair", label="crosshair", density="compact", hide_details=True
                )
                html.Div("x {{ x }}  y {{ y }}  z {{ z }}  zoom {{ Number(zoom).toFixed(2) }}×")
            html.Div("{{ error }}", v_show="error", style="color:#ff8a80;padding:0 12px")
            links = " ".join(
                f'<a href="/?image={case}">{case}</a>' for case in case_ids()
            )
            html.Div(links, v_show="!has_case", style="padding:12px")
            with html.Div(v_show="has_case", style="padding:0 12px 12px"):
                for row in (0, 1):
                    with html.Div(v_show="true" if row == 0 else "row2"):
                        if row == 1:
                            with html.Div(
                                style="display:flex;gap:12px;align-items:center;padding:4px 0"
                            ):
                                vuetify3.VSelect(
                                    v_model="mask2",
                                    items=("mask_items",),
                                    label="Overlay",
                                    density="compact",
                                    hide_details=True,
                                    style="max-width:140px",
                                )
                                vuetify3.VCheckbox(
                                    v_model="overlay2",
                                    label="show",
                                    density="compact",
                                    hide_details=True,
                                )
                        with html.Div(style="display:flex;gap:8px;align-items:stretch"):
                            for plane in PLANES:
                                pane = self.pane(f"{plane}{row}")
                                shown = f"show_{plane}"
                                with html.Div(
                                    v_show=shown, style="flex:1;min-width:0"
                                ):
                                    html.Div(plane)
                                    with html.Div(style="position:relative;background:#000"):
                                        pane.view = vtk_widgets.VtkLocalView(
                                            pane.window,
                                            ref=pane.ref,
                                            interactive_ratio=1,
                                            still_ratio=1,
                                            interactor_settings=("[]",),
                                            interactor_events=events,
                                            LeftButtonPress=(
                                                self._press("cross", pane.ref),
                                                self._ptr(pane.ref),
                                            ),
                                            LeftButtonRelease=(
                                                self._release("cross", pane.ref),
                                                self._ptr(pane.ref),
                                            ),
                                            MiddleButtonPress=(
                                                self._press("zoom", pane.ref),
                                                self._ptr(pane.ref),
                                            ),
                                            MiddleButtonRelease=(
                                                self._release("zoom", pane.ref),
                                                self._ptr(pane.ref),
                                            ),
                                            MouseMove=(
                                                self._move(pane.ref),
                                                self._ptr(pane.ref),
                                            ),
                                            MouseWheel=(
                                                self._wheel(pane.ref),
                                                self._wheel_expr(pane.ref),
                                            ),
                                            resize=(
                                                self._resize(pane.ref),
                                                self._size_expr(pane.ref),
                                            ),
                                            style="width:100%;height:34vh",
                                        )
                                        left, right = self._tags(plane)
                                        html.Span(
                                            left,
                                            style=(
                                                "position:absolute;top:4px;left:6px;color:#3dffa6;"
                                                "pointer-events:none;z-index:2;text-shadow:0 0 3px #000"
                                            ),
                                        )
                                        html.Span(
                                            right,
                                            style=(
                                                "position:absolute;top:4px;right:6px;color:#3dffa6;"
                                                "pointer-events:none;z-index:2;text-shadow:0 0 3px #000"
                                            ),
                                        )
        state.change(
            "case_id",
            "mask",
            "overlay",
            "mask2",
            "overlay2",
            "row2",
            "window_name",
            "opacity",
            "view_name",
            "crosshair",
            "zoom",
            "x",
            "y",
            "z",
            "show_axial",
            "show_coronal",
            "show_sagittal",
        )(self.on_change)

    def _tags(self, plane):
        if plane == "sagittal":
            return "A", "P"
        return (
            "{{ view_name === 'Radiologist' ? 'R' : 'L' }}",
            "{{ view_name === 'Radiologist' ? 'L' : 'R' }}",
        )

    def _canvas(self, ref):
        # Trame stores the view on trame.refs, not $refs.
        return (
            f"(trame.refs['{ref}']&&(trame.refs['{ref}'].$el||trame.refs['{ref}']))"
            f".querySelector('canvas')"
        )

    def _ptr(self, ref):
        canvas = self._canvas(ref)
        return (
            f"($event&&$event.position?[$event.position.x,$event.position.y,"
            f"{canvas}.width,{canvas}.height]:[0,0,1,1])"
        )

    def _wheel_expr(self, ref):
        canvas = self._canvas(ref)
        return (
            f"($event&&$event.position?[$event.position.x,$event.position.y,"
            f"$event.spinY||0,{canvas}.width,{canvas}.height]:[0,0,0,1,1])"
        )

    def _size_expr(self, ref):
        canvas = self._canvas(ref)
        return f"(()=>{{const c={canvas};return c?[c.width,c.height]:[1,1]}})()"

    def _args(self, args):
        # Trame splats the JS array into the handler. A single list is a payload.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return list(args)

    def _press(self, mode, ref):
        def handler(*args):
            self.on_press(mode, ref, self._args(args))

        return handler

    def _release(self, mode, ref):
        def handler(*args):
            self.on_release(mode, ref, self._args(args))

        return handler

    def _move(self, ref):
        def handler(*args):
            self.on_move(ref, self._args(args))

        return handler

    def _wheel(self, ref):
        def handler(*args):
            self.on_wheel(ref, self._args(args))

        return handler

    def _resize(self, ref):
        def handler(*args):
            self.on_resize(ref, self._args(args))

        return handler

    def open_query(self, search, **_kwargs):
        failed = False
        self._focal = None
        self._refreshing = True
        try:
            parsed = parse_search(search or "")
            with state:
                state.error = ""
                if not parsed["case_id"]:
                    state.has_case = False
                    state.address = "/"
                else:
                    for key, value in parsed.items():
                        setattr(state, key, value)
                    state.has_case = True
        except ValueError as exc:
            state.error = str(exc)
            state.has_case = False
            failed = True
        finally:
            self._refreshing = False
        if not failed and state.has_case:
            self.refresh()

    def on_change(self, **_kwargs):
        if self._refreshing or not state.has_case:
            return
        self.refresh()

    def refresh(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.ensure_loaded()
            self.apply_style()
            self.place()
            self.write_address()
            state.error = ""
        except Exception as exc:
            state.error = str(exc)
            state.has_case = False
            return
        finally:
            self._refreshing = False
        self.request_push()

    def ensure_loaded(self):
        if state.window_name not in WINDOWS:
            raise ValueError(f"unknown window: {state.window_name}")
        if state.view_name not in VIEWS:
            raise ValueError(f"unknown view: {state.view_name}")
        image_path, mask_path = lookup(state.case_id, state.mask)
        _image_path, mask2_path = lookup(state.case_id, state.mask2)
        if image_path != self._ct_path:
            _LUTS.clear()
            self._ct_path = image_path
        ct = read_volume(self.ct_reader, image_path)
        mask = read_volume(self.mask_readers[0], mask_path)
        mask2 = read_volume(self.mask_readers[1], mask2_path)
        if ct.GetDimensions() != mask.GetDimensions() or ct.GetDimensions() != mask2.GetDimensions():
            raise ValueError("shape mismatch")
        img_w, img_d, img_h = ct.GetDimensions()
        state.img_w, state.img_d, state.img_h = int(img_w), int(img_d), int(img_h)
        if int(state.x) < 0 or int(state.y) < 0 or int(state.z) < 0:
            state.x, state.y, state.z = focus_of(mask)
        state.x = clamp(state.x, img_w)
        state.y = clamp(state.y, img_d)
        state.z = clamp(state.z, img_h)
        level, width = WINDOWS[state.window_name]
        lut, ctf, pwf = colors_for(mask_path, mask)
        lut2, ctf2, pwf2 = colors_for(mask2_path, mask2)
        colors = {0: (lut, ctf, pwf), 1: (lut2, ctf2, pwf2)}
        for pane in self.panes:
            pane.image_colors.SetInputConnection(self.ct_reader.GetOutputPort())
            pane.image_colors.SetWindow(float(width))
            pane.image_colors.SetLevel(float(level))
            row_lut, row_ctf, row_pwf = colors[pane.row]
            pane.mask_colors.SetInputConnection(self.mask_readers[pane.row].GetOutputPort())
            pane.mask_colors.SetLookupTable(row_lut)
            prop = pane.mask_slice.GetProperty()
            prop.trame_ctf = row_ctf
            prop.trame_pwf = row_pwf
        self._mask_paths = {0: mask_path, 1: mask2_path}

    def apply_style(self):
        if state.window_name not in WINDOWS:
            raise ValueError(f"unknown window: {state.window_name}")
        if state.view_name not in VIEWS:
            raise ValueError(f"unknown view: {state.view_name}")
        level, width = WINDOWS[state.window_name]
        opacity = min(100, max(0, int(round(float(state.opacity)))))
        state.opacity = opacity
        for pane in self.panes:
            prop = pane.image_slice.GetProperty()
            prop.SetColorWindow(float(width))
            prop.SetColorLevel(float(level))
            prop.SetOpacity(1)
            prop.SetInterpolationTypeToNearest()
            prop.SetUseLookupTableScalarRange(False)
            pane.image_colors.SetWindow(float(width))
            pane.image_colors.SetLevel(float(level))
            show = state.overlay if pane.row == 0 else state.overlay2
            alpha = (opacity / 100.0) if show else 0.0
            mask_prop = pane.mask_slice.GetProperty()
            mask_prop.SetOpacity(alpha)
            mask_prop.SetInterpolationTypeToNearest()
            mask_prop.SetUseLookupTableScalarRange(True)
            pane.mask_slice.SetVisibility(1 if alpha > 0 else 0)
            pane.image_slice.SetVisibility(1)
            for _source, actor in pane.lines:
                actor.SetVisibility(1 if state.crosshair else 0)

    def place(self, camera=True):
        ct = self.ct_reader.GetOutput()
        if ct is None or ct.GetDimensions() == (0, 0, 0):
            return
        axes = getattr(self.ct_reader, "loaded_axes", np.eye(3))
        x, y, z = int(state.x), int(state.y), int(state.z)
        zoom = float(state.zoom)
        # The camera center stays put while the crosshair moves, so a click
        # does not pan or zoom the slice.
        if camera and self._focal is None:
            self._focal = (x, y, z)
        spacing = np.array(ct.GetSpacing(), float)
        dims = ct.GetDimensions()
        dist = float(np.max(np.array(dims) * spacing)) * 3 + 1
        index = (x, y, z)
        cross = world_point(ct, x, y, z)
        for pane in self.panes:
            dop, up = screen_basis(axes, pane.plane, state.view_name)
            world_dop = world_vector(ct, dop)
            world_up = world_vector(ct, up)
            world_dop = world_dop / np.linalg.norm(world_dop)
            world_up = world_up / np.linalg.norm(world_up)
            normal = np.zeros(3)
            normal[_NORMAL[pane.plane]] = 1
            world_normal = world_vector(ct, normal)
            pane.slice_plane.SetOrigin(*cross)
            pane.slice_plane.SetNormal(*world_normal)
            slice_index = index[_SLICE_MODE[pane.plane]]
            pane.image_mapper.slice_mode = _SLICE_MODE[pane.plane]
            pane.mask_mapper.slice_mode = _SLICE_MODE[pane.plane]
            pane.image_mapper.slice_index = slice_index
            pane.mask_mapper.slice_index = slice_index
            if camera and self._focal is not None:
                focal = world_point(ct, *self._focal)
                aspect = self._aspect.get(pane.ref, 1.0)
                scale = parallel_scale(ct, pane.plane, zoom, aspect)
                cam = pane.camera
                cam.SetParallelProjection(True)
                cam.SetFocalPoint(*focal)
                cam.SetPosition(*(focal - world_dop * dist))
                cam.SetViewUp(*world_up)
                cam.SetParallelScale(scale)
                cam.SetClippingRange(0.01, dist * 8)
            nudge = -world_dop * float(np.min(spacing)) * 0.5
            for axis, (source, _actor) in zip(_PLANE_AXES[pane.plane], pane.lines):
                start = [x, y, z]
                end = [x, y, z]
                start[axis] = 0
                end[axis] = dims[axis] - 1
                p1 = world_point(ct, *start) + nudge
                p2 = world_point(ct, *end) + nudge
                source.SetPoint1(*p1)
                source.SetPoint2(*p2)
                source.Modified()

    def write_address(self):
        if not state.case_id:
            state.address = "/"
            return
        query = {
            "image": state.case_id,
            "mask": state.mask,
            "overlay": int(bool(state.overlay)),
            "mask2": state.mask2,
            "overlay2": int(bool(state.overlay2)),
            "row2": int(bool(state.row2)),
            "window": state.window_name,
            "opacity": int(state.opacity),
            "view": state.view_name,
            "crosshair": int(bool(state.crosshair)),
            "zoom": f"{float(state.zoom):.2f}",
            "axial": int(bool(state.show_axial)),
            "coronal": int(bool(state.show_coronal)),
            "sagittal": int(bool(state.show_sagittal)),
            "x": int(state.x),
            "y": int(state.y),
            "z": int(state.z),
        }
        state.address = "/?" + urlencode(query)

    def request_push(self):
        self._push_gen += 1
        gen = self._push_gen
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.push()
            return
        loop.call_later(0.04, lambda: self._push_if(gen))

    def _push_if(self, gen):
        if gen == self._push_gen:
            self.push()

    def flush_push(self):
        self._push_gen += 1
        self.push()

    def push(self):
        if not server.protocol:
            return
        install_serializers()
        for pane in self.panes:
            if pane.view is None:
                continue
            pane.view.update()
            pane.view.push_camera(pane.camera)

    def note_size(self, ref, width, height):
        if width and height and float(height) > 0:
            self._aspect[ref] = float(width) / float(height)

    def on_press(self, mode, ref, payload):
        if not state.has_case or not payload:
            return
        self.note_size(ref, payload[2], payload[3])
        if mode == "cross":
            self.aim(ref, payload)
        self._drag = {
            "mode": mode,
            "ref": ref,
            "t": time.time(),
            "y": float(payload[1]),
            "zoom": float(state.zoom),
            "moved": False,
        }

    def on_release(self, mode, ref, _payload):
        drag = self._drag
        if not drag or drag["mode"] != mode or drag["ref"] != ref:
            self.flush_push()
            return
        if mode == "zoom" and not drag["moved"] and (time.time() - drag["t"]) < 0.04:
            return
        self._drag = None
        self.flush_push()

    def on_move(self, ref, payload):
        drag = self._drag
        if not drag or drag["ref"] != ref or not payload:
            return
        if drag["mode"] == "cross":
            self.note_size(ref, payload[2], payload[3])
            self.aim(ref, payload)
        elif drag["mode"] == "zoom":
            dy = float(payload[1]) - drag["y"]
            if abs(dy) > 0:
                drag["moved"] = True
            zoom = drag["zoom"] * math.exp(dy / 160.0)
            self.set_zoom(round(min(32.0, max(1.0, zoom)), 2))

    def on_wheel(self, ref, payload):
        if not state.has_case or not payload or len(payload) < 3:
            return
        spin = float(payload[2])
        if spin == 0:
            return
        step = 1 if spin > 0 else -1
        plane = self.pane(ref).plane
        x, y, z = int(state.x), int(state.y), int(state.z)
        if plane == "axial":
            z += step
        elif plane == "coronal":
            y += step
        else:
            x += step
        self.set_indices(x, y, z)

    def on_resize(self, ref, payload):
        if not payload:
            return
        self.note_size(ref, payload[0], payload[1])
        if state.has_case:
            self.place()
            self.request_push()

    def aim(self, ref, payload):
        px, py, width, height = (float(v) for v in payload[:4])
        if width < 1 or height < 1:
            return
        pane = self.pane(ref)
        ct = self.ct_reader.GetOutput()
        camera = pane.camera
        world = display_world(
            camera.GetFocalPoint(),
            camera.GetDirectionOfProjection(),
            camera.GetViewUp(),
            camera.GetParallelScale(),
            width / height,
            px,
            py,
            width,
            height,
        )
        hit = index_of(ct, world)
        x, y, z = int(state.x), int(state.y), int(state.z)
        if pane.plane == "axial":
            x, y = int(hit[0]), int(hit[1])
        elif pane.plane == "coronal":
            x, z = int(hit[0]), int(hit[2])
        else:
            y, z = int(hit[1]), int(hit[2])
        self.set_indices(x, y, z)

    def set_indices(self, x, y, z):
        self._refreshing = True
        try:
            state.x = clamp(x, state.img_w)
            state.y = clamp(y, state.img_d)
            state.z = clamp(z, state.img_h)
            self.place(camera=False)
            self.write_address()
        finally:
            self._refreshing = False
        self.request_push()

    def set_zoom(self, zoom):
        self._refreshing = True
        try:
            state.zoom = zoom
            self.place()
            self.write_address()
        finally:
            self._refreshing = False
        self.request_push()


viewer = Viewer()


def main():
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = int(os.environ.get("PORT", "80"))
    server.start(host="0.0.0.0", port=port, open_browser=False)


if __name__ == "__main__":
    main()
