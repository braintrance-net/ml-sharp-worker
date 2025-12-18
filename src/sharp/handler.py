"""RunPod serverless handler for SHARP image-to-PLY conversion."""

import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
import runpod
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply

from sharp.cli.predict import predict_image

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Environment variables
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "braintrance-mlsharp-bucket")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "mlsharp/")
CLOUDFRONT_URL = os.environ.get("CLOUDFRONT_URL", "https://dnht1hs3nve3d.cloudfront.net")
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
LOCAL_MODEL_PATH = "/app/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"

# Global model (loaded once)
MODEL = None
DEVICE = None


def load_model():
    """Load the SHARP model (cached globally)."""
    global MODEL, DEVICE

    if MODEL is not None:
        return MODEL, DEVICE

    LOGGER.info("Loading SHARP model...")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    LOGGER.info(f"Using device: {DEVICE}")

    # Load from local file if exists, otherwise download
    if os.path.exists(LOCAL_MODEL_PATH):
        LOGGER.info(f"Loading model from local cache: {LOCAL_MODEL_PATH}")
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE, weights_only=True)
    else:
        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    MODEL = create_predictor(PredictorParams())
    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    MODEL.to(DEVICE)

    LOGGER.info("Model loaded successfully")
    return MODEL, DEVICE


def download_image(url: str, dest_path: Path) -> Path:
    """Download image from URL (supports HTTP/HTTPS and S3 URLs)."""
    if url.startswith("s3://"):
        # Parse s3://bucket/key
        parts = url[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        LOGGER.info(f"Downloading from S3: bucket={bucket}, key={key}")
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, str(dest_path))
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
    return dest_path


def upload_to_s3(local_path: Path, s3_key: str) -> tuple[str, str]:
    """Upload file to S3 and return (s3_path, cloudfront_url)."""
    s3_client = boto3.client("s3")

    s3_client.upload_file(
        str(local_path),
        OUTPUT_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "application/octet-stream"}
    )

    s3_path = f"s3://{OUTPUT_BUCKET}/{s3_key}"
    cloudfront_url = f"{CLOUDFRONT_URL}/{s3_key}"
    return s3_path, cloudfront_url


def handler(job):
    """RunPod handler for SHARP prediction."""
    job_input = job["input"]

    # Get input image (URL or base64)
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    job_id = job_input.get("job_id", str(uuid.uuid4()))

    if not image_url and not image_base64:
        return {"error": "Must provide either 'image_url' or 'image_base64'"}

    try:
        # Load model
        model, device = load_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Get input image
            if image_url:
                parsed = urlparse(image_url)
                ext = os.path.splitext(parsed.path)[1] or ".jpg"
                image_path = tmpdir / f"input{ext}"
                LOGGER.info(f"Downloading image from {image_url}")
                download_image(image_url, image_path)
            else:
                image_path = tmpdir / "input.jpg"
                LOGGER.info("Decoding base64 image")
                image_data = base64.b64decode(image_base64)
                image_path.write_bytes(image_data)

            # Load and process image
            LOGGER.info(f"Processing image: {image_path}")
            image, _, f_px = io.load_rgb(image_path)
            height, width = image.shape[:2]

            # Run prediction
            LOGGER.info("Running SHARP prediction...")
            gaussians = predict_image(model, image, f_px, device)

            # Save PLY locally
            local_ply_path = tmpdir / f"{job_id}.ply"
            save_ply(gaussians, f_px, (height, width), local_ply_path)
            LOGGER.info(f"Saved PLY to {local_ply_path}")

            # Upload to S3
            output_key = f"{OUTPUT_PREFIX}{job_id}.ply"
            LOGGER.info(f"Uploading to s3://{OUTPUT_BUCKET}/{output_key}")
            s3_path, public_url = upload_to_s3(local_ply_path, output_key)
            LOGGER.info(f"Uploaded to {public_url}")

            return {
                "status": "success",
                "job_id": job_id,
                "output_s3_path": s3_path,
                "public_url": public_url,
                "image_size": {"width": width, "height": height}
            }

    except Exception as e:
        LOGGER.error(f"Error processing job: {e}", exc_info=True)
        return {"error": str(e)}


# Start RunPod serverless
runpod.serverless.start({"handler": handler})
