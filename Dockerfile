FROM python:3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 build-essential make vim curl wget \
    && rm -rf /var/lib/apt/lists/*

# OSMesa is the offscreen window. Install it last so trame's vtk wheel
# does not replace the OSMesa vtkmodules package.
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall vtk-osmesa \
        --extra-index-url https://gitlab.kitware.com/api/v4/projects/13/packages/pypi/simple

COPY app.py cases.csv ./

EXPOSE 80
CMD ["python", "app.py"]
