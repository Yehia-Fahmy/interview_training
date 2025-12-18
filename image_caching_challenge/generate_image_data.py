"""
Image Data Generator for Image Caching Challenge

This script generates test images and metadata for the image caching interview challenge.

Usage:
    python generate_image_data.py

This will create:
    - images/ directory with sample PNG and JPEG images
    - image_metadata.csv with image details
    - access_log.csv with simulated access patterns
"""

import os
import csv
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Try to import PIL, fall back to creating simple binary files if not available
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed. Will create placeholder binary files instead.")
    print("Install with: pip install Pillow")


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("images")
METADATA_FILE = "image_metadata.csv"
ACCESS_LOG_FILE = "access_log.csv"

# Image categories and their properties
IMAGE_CATEGORIES = {
    "map_tile": {
        "count": 50,
        "size_range": (256, 256),  # Fixed size for map tiles
        "formats": ["png"],
        "colors": ["#4A90D9", "#5BA55B", "#E8E8E8", "#F5DEB3"],  # Blue, green, gray, tan
    },
    "driver_photo": {
        "count": 20,
        "size_range": (100, 150),
        "formats": ["jpeg"],
        "colors": ["#FFB6C1", "#87CEEB", "#98FB98", "#DDA0DD"],
    },
    "vehicle": {
        "count": 15,
        "size_range": (200, 300),
        "formats": ["png", "jpeg"],
        "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
    },
    "banner": {
        "count": 10,
        "size_range": (600, 800),
        "formats": ["png", "jpeg"],
        "colors": ["#667eea", "#764ba2", "#f093fb", "#f5576c"],
    },
}

# Access pattern configuration
NUM_ACCESS_EVENTS = 1000
POPULAR_IMAGE_RATIO = 0.2  # Top 20% of images get 80% of accesses


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageMetadata:
    """Metadata for a generated image."""
    image_id: str
    filename: str
    format: str
    width: int
    height: int
    size_bytes: int
    category: str
    popularity_score: float  # 0.0 to 1.0


# =============================================================================
# Image Generation
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_gradient_image(
    width: int,
    height: int,
    color1: str,
    color2: str
) -> Image.Image:
    """Generate a gradient image with some visual elements."""
    img = Image.new('RGB', (width, height))
    
    c1 = hex_to_rgb(color1)
    c2 = hex_to_rgb(color2)
    
    # Create gradient
    for y in range(height):
        ratio = y / height
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))
    
    # Add some visual elements
    draw = ImageDraw.Draw(img)
    
    # Add random circles/shapes
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, width - 20)
        y1 = random.randint(0, height - 20)
        x2 = x1 + random.randint(10, 50)
        y2 = y1 + random.randint(10, 50)
        
        fill_color = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )
        draw.ellipse([x1, y1, x2, y2], fill=fill_color, outline=None)
    
    return img


def generate_map_tile_image(width: int, height: int, colors: List[str]) -> Image.Image:
    """Generate a map tile-like image with grid pattern."""
    img = Image.new('RGB', (width, height), color=hex_to_rgb(colors[2]))  # Background
    draw = ImageDraw.Draw(img)
    
    # Draw grid lines (roads)
    road_color = (200, 200, 200)
    for x in range(0, width, 32):
        draw.line([(x, 0), (x, height)], fill=road_color, width=2)
    for y in range(0, height, 32):
        draw.line([(0, y), (width, y)], fill=road_color, width=2)
    
    # Add some "buildings" (rectangles)
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, width - 30)
        y1 = random.randint(0, height - 30)
        x2 = x1 + random.randint(15, 30)
        y2 = y1 + random.randint(15, 30)
        
        building_color = hex_to_rgb(random.choice(colors[:2]))
        draw.rectangle([x1, y1, x2, y2], fill=building_color)
    
    return img


def generate_placeholder_bytes(size: int, image_id: str) -> bytes:
    """Generate placeholder binary data when PIL is not available."""
    # Create deterministic but varied data based on image_id
    seed = hashlib.md5(image_id.encode()).hexdigest()
    random.seed(seed)
    
    # Create a simple binary header + random-ish data
    header = f"IMAGE:{image_id}:".encode()
    data_size = size - len(header)
    data = bytes([random.randint(0, 255) for _ in range(data_size)])
    
    return header + data


def generate_image(
    category: str,
    index: int,
    config: Dict
) -> Tuple[ImageMetadata, bytes]:
    """Generate a single image and its metadata."""
    # Determine image properties
    size_range = config["size_range"]
    width = random.randint(size_range[0], size_range[1])
    height = random.randint(size_range[0], size_range[1])
    
    img_format = random.choice(config["formats"])
    image_id = f"{category}_{index:03d}"
    filename = f"{image_id}.{img_format}"
    
    # Assign popularity score (exponential distribution)
    popularity_score = random.expovariate(2.0)
    popularity_score = min(1.0, popularity_score)
    
    if HAS_PIL:
        # Generate actual image
        colors = config["colors"]
        
        if category == "map_tile":
            img = generate_map_tile_image(width, height, colors)
        else:
            color1 = random.choice(colors)
            color2 = random.choice(colors)
            img = generate_gradient_image(width, height, color1, color2)
        
        # Save to bytes
        import io
        buffer = io.BytesIO()
        
        if img_format == "jpeg":
            img = img.convert("RGB")  # Ensure RGB for JPEG
            img.save(buffer, format="JPEG", quality=85)
        else:
            img.save(buffer, format="PNG")
        
        image_bytes = buffer.getvalue()
        
        # Get actual dimensions after any processing
        width, height = img.size
    else:
        # Create placeholder binary data
        estimated_size = width * height * 3  # Rough estimate
        image_bytes = generate_placeholder_bytes(estimated_size, image_id)
    
    metadata = ImageMetadata(
        image_id=image_id,
        filename=filename,
        format=img_format,
        width=width,
        height=height,
        size_bytes=len(image_bytes),
        category=category,
        popularity_score=popularity_score,
    )
    
    return metadata, image_bytes


# =============================================================================
# Access Pattern Generation
# =============================================================================

def generate_access_log(
    metadata_list: List[ImageMetadata],
    num_events: int = NUM_ACCESS_EVENTS
) -> List[Dict]:
    """Generate simulated access log following Zipf-like distribution."""
    # Sort by popularity to create skewed access pattern
    sorted_images = sorted(
        metadata_list,
        key=lambda x: x.popularity_score,
        reverse=True
    )
    
    popular_count = max(1, int(len(sorted_images) * POPULAR_IMAGE_RATIO))
    popular_images = sorted_images[:popular_count]
    other_images = sorted_images[popular_count:]
    
    access_log = []
    current_time = 0.0
    
    for i in range(num_events):
        # 80% of accesses go to popular images
        if random.random() < 0.8 and popular_images:
            image = random.choice(popular_images)
        else:
            image = random.choice(sorted_images)
        
        # Increment time (variable intervals)
        current_time += random.expovariate(10.0)  # ~100ms average
        
        access_log.append({
            "timestamp": round(current_time, 3),
            "image_id": image.image_id,
            "category": image.category,
            "size_bytes": image.size_bytes,
            "is_popular": image in popular_images,
        })
    
    return access_log


# =============================================================================
# File I/O
# =============================================================================

def save_metadata_csv(metadata_list: List[ImageMetadata], filepath: str) -> None:
    """Save image metadata to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "filename", "format", "width", "height",
            "size_bytes", "category", "popularity_score"
        ])
        
        for meta in metadata_list:
            writer.writerow([
                meta.image_id,
                meta.filename,
                meta.format,
                meta.width,
                meta.height,
                meta.size_bytes,
                meta.category,
                f"{meta.popularity_score:.4f}",
            ])


def save_access_log_csv(access_log: List[Dict], filepath: str) -> None:
    """Save access log to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_id", "category", "size_bytes", "is_popular"])
        
        for event in access_log:
            writer.writerow([
                event["timestamp"],
                event["image_id"],
                event["category"],
                event["size_bytes"],
                event["is_popular"],
            ])


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Generate all test data."""
    print("=" * 60)
    print("Image Data Generator for Caching Challenge")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    
    # Generate images
    all_metadata = []
    total_size = 0
    
    for category, config in IMAGE_CATEGORIES.items():
        print(f"\nGenerating {config['count']} {category} images...")
        
        for i in range(config['count']):
            metadata, image_bytes = generate_image(category, i, config)
            
            # Save image file
            filepath = OUTPUT_DIR / metadata.filename
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            all_metadata.append(metadata)
            total_size += metadata.size_bytes
        
        print(f"  Created {config['count']} images")
    
    # Save metadata CSV
    save_metadata_csv(all_metadata, METADATA_FILE)
    print(f"\nSaved metadata to: {METADATA_FILE}")
    
    # Generate and save access log
    print(f"\nGenerating access log with {NUM_ACCESS_EVENTS} events...")
    access_log = generate_access_log(all_metadata)
    save_access_log_csv(access_log, ACCESS_LOG_FILE)
    print(f"Saved access log to: {ACCESS_LOG_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total images: {len(all_metadata)}")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Categories: {', '.join(IMAGE_CATEGORIES.keys())}")
    print(f"  Access events: {len(access_log)}")
    
    # Category breakdown
    print(f"\nImages by category:")
    for category in IMAGE_CATEGORIES:
        count = sum(1 for m in all_metadata if m.category == category)
        cat_size = sum(m.size_bytes for m in all_metadata if m.category == category)
        print(f"  {category}: {count} images ({cat_size / 1024:.1f} KB)")
    
    print(f"\nFiles created:")
    print(f"  - {OUTPUT_DIR}/ (image files)")
    print(f"  - {METADATA_FILE}")
    print(f"  - {ACCESS_LOG_FILE}")
    
    print("\nRun 'python starter_image_caching.py' to test your implementation.")


if __name__ == "__main__":
    main()
