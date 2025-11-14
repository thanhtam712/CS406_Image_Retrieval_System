import os
import json
import base64
import shutil
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import torch
import uvicorn

from main import process_workflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Animal Detection & Retrieval System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# Resolve project root and static/uploads paths so server works when started from `src/`
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
CONFIG = {
    "output_dir": "synthetic_images/cropped_animals",
    "model_path": "models/best_animal_classifier.pth",
    "class_names_path": "models/class_names.json",
    "summaries_path": "animal_summaries.json",
    "model_name": "ViT-H-14",
    "pretrained": "laion2b_s32b_b79k",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "top_k": 10,
    "keep_cropped": True,
    "skip_detection": True  # Set to True to skip object detection
}


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail=f"Frontend not found at {index_path}")
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": CONFIG["device"],
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return JSONResponse(
        content={
            "success": True,
            "message": "Test successful",
            "objects": [
                {
                    "object_id": 0,
                    "cropped_image": None,
                    "predicted_class": "test_animal",
                    "confidence": 0.99,
                    "summary": "This is a test summary",
                    "similar_images": [
                        {"score": 0.95, "metadata": "test1.jpg", "path": "test1.jpg"},
                        {"score": 0.90, "metadata": "test2.jpg", "path": "test2.jpg"}
                    ]
                }
            ]
        }
    )


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image and get animal detection, classification, and retrieval results
    """
    logger.info(f"Received upload request: {file.filename}")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    logger.info(f"Saving file to: {file_path}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify file exists and has content
        if not file_path.exists():
            raise Exception(f"File was not saved: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise Exception(f"File is empty: {file_path}")
        
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    try:
        # Double check file still exists before processing
        if not file_path.exists():
            raise Exception(f"File disappeared before processing: {file_path}")
        
        logger.info(f"Starting image processing...")
        
        # Process the image
        # Convert configured relative paths to absolute paths under project root
        result = process_workflow(
            input_image=str(file_path),
            output_dir=str(BASE_DIR / CONFIG["output_dir"]),
            model_path=str(BASE_DIR / CONFIG["model_path"]),
            class_names_path=str(BASE_DIR / CONFIG["class_names_path"]),
            summaries_path=str(BASE_DIR / CONFIG["summaries_path"]),
            model_name=CONFIG["model_name"],
            pretrained=CONFIG["pretrained"],
            device=CONFIG["device"],
            top_k=CONFIG["top_k"],
            keep_cropped=CONFIG["keep_cropped"],
            skip_detection=CONFIG["skip_detection"]
        )
        
        logger.info(f"Processing completed. Found animals: {result is not None}")
        
        if result is None:
            logger.info("No animals detected")
            return JSONResponse(
                content={
                    "success": False,
                    "message": "No animals detected in the image",
                    "objects": []
                }
            )
        
        imgs_retrieved, predict_class, conf, summ = result
        
        # Format response
        objects = []
        for idx, (images, predicted_class, confidence, summary) in enumerate(
            zip(imgs_retrieved, predict_class, conf, summ)
        ):
            cropped_image_b64 = None
            
            # If skip_detection is enabled, use the original uploaded image
            if CONFIG["skip_detection"]:
                logger.info(f"Using original uploaded image (skip_detection=True)")
                try:
                    with open(file_path, "rb") as img_file:
                        cropped_image_b64 = base64.b64encode(img_file.read()).decode()
                    logger.info(f"Encoded original image, size: {len(cropped_image_b64)} chars")
                except Exception as e:
                    logger.error(f"Failed to encode original image: {e}")
            else:
                # Get cropped image path (use absolute path)
                cropped_image_path = BASE_DIR / CONFIG["output_dir"] / f"crop_{idx}.jpg"
                logger.info(f"Looking for cropped image at: {cropped_image_path}")
                
                if cropped_image_path.exists():
                    try:
                        with open(cropped_image_path, "rb") as img_file:
                            cropped_image_b64 = base64.b64encode(img_file.read()).decode()
                        logger.info(f"Encoded cropped image {idx}, size: {len(cropped_image_b64)} chars")
                    except Exception as e:
                        logger.error(f"Failed to encode cropped image {idx}: {e}")
                else:
                    logger.warning(f"Cropped image not found: {cropped_image_path}")
            
            # Format retrieved images
            similar_images = []
            for result in images:
                metadata = result.entity.get('metadata', '')
                # Handle metadata - it might be a dict or string
                if isinstance(metadata, dict):
                    file_path = metadata.get('file_path', '')
                else:
                    file_path = metadata
                
                similar_images.append({
                    "score": float(result.distance),
                    "metadata": file_path,
                    "path": file_path
                })
            
            objects.append({
                "object_id": idx,
                "cropped_image": cropped_image_b64,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "summary": summary,
                "similar_images": similar_images
            })
        
        logger.info(f"Formatted {len(objects)} animal objects for response")
        
        # Calculate response size
        response_data = {
            "success": True,
            "message": f"Detected {len(objects)} animal(s)",
            "objects": objects
        }
        
        response_json = json.dumps(response_data)
        response_size = len(response_json)
        logger.info(f"Response size: {response_size:,} bytes ({response_size/1024:.1f} KB)")
        
        if response_size > 10_000_000:  # 10MB limit
            logger.warning(f"Response size ({response_size/1024/1024:.1f} MB) is very large!")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error during image processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")


@app.get("/data/{image_path:path}")
async def get_image(image_path: str):
    """
    Serve images from the data directory
    """
    full_path = BASE_DIR / "data" / image_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(full_path))


# Mount frontend static files (use absolute path)
if not STATIC_DIR.exists():
    # If static dir is missing, mount a no-op to avoid crash and provide a helpful error when root is accessed
    raise RuntimeError(f"Directory '{STATIC_DIR}' does not exist")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
