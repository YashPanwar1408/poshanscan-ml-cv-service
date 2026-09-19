# PoshanScan CV Service

Computer-vision microservice that estimates **MUAC** (Mid-Upper Arm Circumference) from a photo of a child’s upper arm. A reference marker (ArUco, known physical size) is placed next to the arm so the pipeline can convert pixels to millimetres. The estimate is intended for child malnutrition screening.

## Pipeline

1. **Reference detection** (`app/pipeline/reference_detect.py`) — find the ArUco marker and compute pixels per millimetre.
2. **Pose localisation** (`app/pipeline/pose_localize.py`) — find the mid-upper-arm measurement site (MediaPipe Pose).
3. **Segmentation** (`app/pipeline/segment.py`) — isolate the arm in a crop around that site (HSV + GrabCut baseline).
4. **Estimation** (`app/pipeline/estimate.py`) — convert apparent width to millimetres, then circumference (`π × shape_factor × width`).
5. **Classification** (`app/pipeline/classify.py`) — WHO bands: Normal (≥125 mm), MAM (115–124.9 mm), SAM (<115 mm).

## Install dependencies

Use Python 3.10+ (3.11 recommended). From this directory:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally with Uvicorn

From the `poshanscan-cv-service` directory (so the `app` package is importable):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

- Health check: `GET http://127.0.0.1:8001/health` → `{"status":"ok"}`
- Interactive docs (Swagger UI): [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
- Inference: `POST /infer`

Each pipeline stage logs an INFO line to the console (marker scale, pose midpoint, mask width, MUAC, risk band).

## Test POST /infer

The endpoint accepts **multipart form-data**:

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | Photo of the upper arm + ArUco marker |
| `reference_type` | string | `aruco` | Reference object type |
| `reference_size_mm` | float | `50.0` | Printed marker edge length in millimetres |

### Swagger UI

1. Open [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
2. Expand **POST /infer** → **Try it out**
3. Choose an image file, optionally set `reference_size_mm`, then **Execute**

A successful response looks like:

```json
{
  "muac_estimate_mm": 142.3,
  "risk_band": "Normal",
  "confidence_score": 0.81,
  "quality_flags": {
    "reference_detected": true,
    "arm_detected": true,
    "segmentation_ok": true
  },
  "pipeline_version": "v1.0"
}
```

If the marker or arm cannot be found, the service returns **HTTP 422** with:

```json
{
  "error": "reference_object_not_detected",
  "message": "No ArUco reference marker was found in the image."
}
```

(`arm_not_detected` is used when pose localisation fails.)

### curl

Linux / macOS:

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "image=@sample_images/arm.jpg" \
  -F "reference_type=aruco" \
  -F "reference_size_mm=50.0"
```

Windows (PowerShell):

```powershell
curl.exe -X POST "http://127.0.0.1:8001/infer" `
  -F "image=@sample_images/arm.jpg" `
  -F "reference_type=aruco" `
  -F "reference_size_mm=50.0"
```

## Docker

```bash
docker build -t poshanscan-cv-service .
docker run --rm -p 8001:8001 poshanscan-cv-service
```

Then use the same `/health` and `/infer` URLs on port **8001**.
