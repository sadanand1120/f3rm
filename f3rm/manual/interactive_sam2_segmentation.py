#!/usr/bin/env python3
"""
Interactive SAM2 Segmentation Tool

Allows users to interactively select points on an image for segmentation using SAM2.
- Left click: Add positive point (foreground)
- Right click: Add negative point (background) 
- Press 'q': Finish selection and save mask
- Press 'r': Reset all points
- Press 'u': Undo last point

Usage:
    python interactive_sam2_segmentation.py --image_path /path/to/image.jpg
"""

import argparse
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Colored logging utilities
class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_colored(text, color=Colors.WHITE, bold=False):
    """Print colored text to terminal"""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.END}")


def print_status(text, status_type="info"):
    """Print status message with appropriate color"""
    if status_type == "success":
        print_colored(f"✓ {text}", Colors.GREEN, bold=True)
    elif status_type == "warning":
        print_colored(f"⚠ {text}", Colors.YELLOW, bold=True)
    elif status_type == "error":
        print_colored(f"✗ {text}", Colors.RED, bold=True)
    elif status_type == "info":
        print_colored(f"ℹ {text}", Colors.BLUE)
    elif status_type == "processing":
        print_colored(f"⟳ {text}", Colors.CYAN)
    else:
        print(text)


class InteractiveSAM2Segmenter:
    def __init__(self, sam2_checkpoint, model_cfg, device):
        """Initialize the SAM2 model and predictor"""
        print_status(f"Loading SAM2 model on device: {device}", "processing")
        start_time = time.time()

        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.device = device

        load_time = time.time() - start_time
        print_status(f"SAM2 model loaded successfully in {load_time:.2f}s", "success")

        # Point storage
        self.points = []
        self.labels = []

        # Display variables
        self.image = None
        self.display_image = None
        self.window_name = "Interactive SAM2 Instance Segmentation"
        self.current_mask = None

        # Instance segmentation variables
        self.instance_mask = None  # Full instance segmentation mask
        self.current_instance_id = 1
        self.instance_colors = {}  # Store colors for each instance

        # Colors for visualization
        self.pos_color = (0, 255, 0)  # Green for positive points
        self.neg_color = (0, 0, 255)  # Red for negative points
        self.marker_size = 8

    def load_image(self, image_path):
        """Load and prepare image for segmentation"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print_status(f"Loading image: {os.path.basename(image_path)}", "processing")

        # Load image
        pil_image = Image.open(image_path)
        self.image = np.array(pil_image.convert("RGB"))

        print_status("Computing image embeddings...", "processing")
        start_time = time.time()

        # Set image in predictor
        self.predictor.set_image(self.image)

        embed_time = time.time() - start_time
        print_status(f"Image embeddings computed in {embed_time:.2f}s", "success")

        # Calculate display size (max HD while maintaining aspect ratio)
        self.display_height, self.display_width = self.image.shape[:2]
        max_display_height = 720
        max_display_width = 1280

        if self.display_height > max_display_height or self.display_width > max_display_width:
            scale_h = max_display_height / self.display_height
            scale_w = max_display_width / self.display_width
            self.display_scale = min(scale_h, scale_w)
            self.display_height = int(self.display_height * self.display_scale)
            self.display_width = int(self.display_width * self.display_scale)
        else:
            self.display_scale = 1.0

        # Initialize instance segmentation mask
        self.instance_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Create display copy (BGR for cv2) and resize for display
        display_img = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2BGR)
        self.display_image = cv2.resize(display_img, (self.display_width, self.display_height))

        print_status(f"Image shape: {self.image.shape}", "info")
        print_status(f"Display size: {self.display_width}x{self.display_height} (scale: {self.display_scale:.2f})", "info")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add positive point
            # Convert display coordinates to original image coordinates
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)
            self.points.append([orig_x, orig_y])
            self.labels.append(1)
            print_colored(f"➕ Added positive point at ({orig_x}, {orig_y})", Colors.GREEN)
            self.update_segmentation()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - add negative point
            # Convert display coordinates to original image coordinates
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)
            self.points.append([orig_x, orig_y])
            self.labels.append(0)
            print_colored(f"➖ Added negative point at ({orig_x}, {orig_y})", Colors.RED)
            self.update_segmentation()

    def update_segmentation(self):
        """Update segmentation based on current points"""
        if len(self.points) == 0:
            self.current_mask = None
            # Create visualization with existing instances
            self.display_image = self.create_visualization_with_instances()
        else:
            print_status("Running SAM2 segmentation...", "processing")
            start_time = time.time()

            # Convert to numpy arrays
            point_coords = np.array(self.points)
            point_labels = np.array(self.labels)

            # Predict with SAM2
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            # Use the best mask
            self.current_mask = masks[0]

            seg_time = time.time() - start_time
            mask_pixels = np.sum(self.current_mask)
            total_pixels = self.current_mask.size
            coverage = (mask_pixels / total_pixels) * 100

            print_status(f"Segmentation completed in {seg_time:.3f}s", "success")
            print_colored(f"   Mask coverage: {coverage:.1f}% ({mask_pixels:,} pixels)", Colors.CYAN)

            # Create visualization with current mask and existing instances
            self.display_image = self.create_visualization_with_instances(point_coords, point_labels)

        # Update display
        cv2.imshow(self.window_name, self.display_image)

    def generate_instance_color(self, instance_id):
        """Generate a unique color for each instance"""
        if instance_id not in self.instance_colors:
            # Generate distinct colors using HSV color space
            hue = (instance_id * 137.5) % 360  # Golden angle for good distribution
            saturation = 0.7 + (instance_id % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (instance_id % 2) * 0.2  # Vary brightness slightly

            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            color = (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
            self.instance_colors[instance_id] = color

        return self.instance_colors[instance_id]

    def create_visualization_with_instances(self, current_points=None, current_labels=None):
        """Create visualization with all instances and current mask"""
        # Start with original image (BGR)
        vis_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2BGR)

        # Create overlay for existing instances
        instance_overlay = np.zeros_like(vis_image)

        # Draw existing instances
        for instance_id in range(1, self.current_instance_id):
            instance_mask = (self.instance_mask == instance_id)
            if np.any(instance_mask):
                color = self.generate_instance_color(instance_id)
                instance_overlay[instance_mask] = color

        # Draw current mask being annotated (if exists)
        if self.current_mask is not None:
            current_color = self.generate_instance_color(self.current_instance_id)
            # Make current mask slightly more transparent to distinguish it
            current_overlay = np.zeros_like(vis_image)
            current_overlay[self.current_mask.astype(bool)] = current_color

            # Blend current mask with a different alpha
            alpha_current = 0.5
            vis_image = cv2.addWeighted(vis_image, 1 - alpha_current, current_overlay, alpha_current, 0)

        # Blend existing instances
        if np.any(instance_overlay):
            alpha_instances = 0.3
            vis_image = cv2.addWeighted(vis_image, 1 - alpha_instances, instance_overlay, alpha_instances, 0)

        # Draw contours for all instances
        for instance_id in range(1, self.current_instance_id):
            instance_mask = (self.instance_mask == instance_id).astype(np.uint8)
            if np.any(instance_mask):
                contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 1)

        # Draw current mask contours
        if self.current_mask is not None:
            contours, _ = cv2.findContours(self.current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)

        # Resize visualization for display
        vis_image_resized = cv2.resize(vis_image, (self.display_width, self.display_height))

        # Draw current annotation points
        if current_points is not None and current_labels is not None:
            for point, label in zip(current_points, current_labels):
                # Convert original coordinates to display coordinates
                display_x = int(point[0] * self.display_scale)
                display_y = int(point[1] * self.display_scale)
                color = self.pos_color if label == 1 else self.neg_color
                marker_size = max(4, int(self.marker_size * self.display_scale))
                cv2.circle(vis_image_resized, (display_x, display_y), marker_size, color, -1)
                cv2.circle(vis_image_resized, (display_x, display_y), marker_size, (255, 255, 255), 2)

        return vis_image_resized

    def save_current_instance(self):
        """Save current mask as an instance and move to next"""
        if self.current_mask is None:
            print_status("No current mask to save as instance!", "warning")
            return

        # Make instance mutually exclusive - remove overlap with existing instances
        current_mask_bool = self.current_mask.astype(bool)
        existing_mask_bool = self.instance_mask > 0

        # Only keep parts of current mask that don't overlap with existing instances
        exclusive_mask = current_mask_bool & ~existing_mask_bool

        if not np.any(exclusive_mask):
            print_status("Current mask completely overlaps with existing instances!", "warning")
            return

        # Add exclusive mask to instance segmentation
        self.instance_mask[exclusive_mask] = self.current_instance_id

        overlap_pixels = np.sum(current_mask_bool & existing_mask_bool)
        final_pixels = np.sum(exclusive_mask)

        print_status(f"Saved instance {self.current_instance_id}", "success")
        print_colored(f"   Instance {self.current_instance_id} pixels: {final_pixels:,}", Colors.CYAN)
        if overlap_pixels > 0:
            print_colored(f"   Removed {overlap_pixels:,} overlapping pixels", Colors.YELLOW)

        # Move to next instance
        self.current_instance_id += 1

        # Clear current annotation
        self.points = []
        self.labels = []
        self.current_mask = None

        print_colored(f"🆕 Ready to annotate instance {self.current_instance_id}", Colors.MAGENTA, bold=True)

        # Update display
        self.update_segmentation()

    def reset_points(self):
        """Reset current annotation points"""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.update_segmentation()
        print_colored("🔄 Reset current annotation points", Colors.YELLOW, bold=True)

    def undo_last_point(self):
        """Remove the last added point"""
        if len(self.points) > 0:
            removed_point = self.points.pop()
            removed_label = self.labels.pop()
            point_type = "positive" if removed_label == 1 else "negative"
            point_color = Colors.GREEN if removed_label == 1 else Colors.RED
            print_colored(f"↶ Undid last {point_type} point at {removed_point}", point_color)
            self.update_segmentation()
        else:
            print_status("No points to undo", "warning")

    def save_instance_mask(self, output_path):
        """Save the instance segmentation mask"""
        if self.current_instance_id == 1:
            print_status("No instances to save!", "error")
            return False

        print_status("Saving instance segmentation mask...", "processing")

        # Save instance mask as numpy array
        np.save(output_path, self.instance_mask)
        print_status(f"Instance mask saved to: {output_path}", "success")

        # Create colored visualization
        self.show_final_segmentation()

        return True

    def show_final_segmentation(self):
        """Display the final instance segmentation with different colors"""
        print_status("Displaying final instance segmentation...", "info")

        # Create colored visualization
        colored_mask = np.zeros((*self.instance_mask.shape, 3), dtype=np.uint8)

        # Color each instance
        for instance_id in range(1, self.current_instance_id):
            instance_pixels = (self.instance_mask == instance_id)
            if np.any(instance_pixels):
                color = self.generate_instance_color(instance_id)
                # Convert BGR to RGB for matplotlib
                rgb_color = (color[2], color[1], color[0])
                colored_mask[instance_pixels] = rgb_color

        # Display results - only 2 panels now
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(colored_mask)
        plt.title("Instance Segmentation")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Print statistics
        total_instances = self.current_instance_id - 1
        print_colored(f"📊 Final Statistics:", Colors.CYAN, bold=True)
        print_colored(f"   Total instances: {total_instances}", Colors.CYAN)
        for instance_id in range(1, self.current_instance_id):
            instance_pixels = np.sum(self.instance_mask == instance_id)
            if instance_pixels > 0:
                print_colored(f"   Instance {instance_id}: {instance_pixels:,} pixels", Colors.CYAN)

    def run_interactive_segmentation(self, image_path):
        """Main interactive segmentation loop"""
        # Load image
        self.load_image(image_path)

        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Initial display
        cv2.imshow(self.window_name, self.display_image)

        print_colored("\n" + "=" * 60, Colors.MAGENTA, bold=True)
        print_colored("Interactive SAM2 Instance Segmentation", Colors.MAGENTA, bold=True)
        print_colored("=" * 60, Colors.MAGENTA, bold=True)
        print_colored("Controls:", Colors.WHITE, bold=True)
        print_colored("  Left click:  Add positive point (green)", Colors.GREEN)
        print_colored("  Right click: Add negative point (red)", Colors.RED)
        print_colored("  'n' key:     Save current instance & start next", Colors.MAGENTA)
        print_colored("  'r' key:     Reset current annotation", Colors.YELLOW)
        print_colored("  'u' key:     Undo last point", Colors.CYAN)
        print_colored("  'q' key:     Finish and save instance mask", Colors.BLUE)
        print_colored("  ESC key:     Exit without saving", Colors.WHITE)
        print_colored("=" * 60, Colors.MAGENTA, bold=True)
        print_colored(f"🎯 Currently annotating instance {self.current_instance_id}", Colors.MAGENTA, bold=True)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit and save instance mask
                if self.current_instance_id > 1:
                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = f"{base_name}_instances.npy"

                    cv2.destroyAllWindows()

                    if self.save_instance_mask(output_path):
                        print_status("Instance segmentation completed successfully!", "success")
                        return output_path
                else:
                    print_status("No instances to save. Annotate some objects first!", "warning")

            elif key == ord('n'):
                # Save current instance and move to next
                self.save_current_instance()

            elif key == ord('r'):
                # Reset current annotation
                self.reset_points()

            elif key == ord('u'):
                # Undo last point
                self.undo_last_point()

            elif key == 27:  # ESC key
                # Exit without saving
                print_status("Exiting without saving...", "warning")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 Segmentation Tool")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--sam2_checkpoint",
                        default="//robodata/smodak/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg",
                        default="//robodata/smodak/repos/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 model config")

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable optimizations for CUDA
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU (will be slow)")

    print_status(f"Using device: {device}", "info")

    # Verify files exist
    # if not os.path.exists(args.sam2_checkpoint):
    #     raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    # if not os.path.exists(args.model_cfg):
    #     raise FileNotFoundError(f"Model config not found: {args.model_cfg}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    # Create segmenter and run
    segmenter = InteractiveSAM2Segmenter(args.sam2_checkpoint, args.model_cfg, device)
    output_path = segmenter.run_interactive_segmentation(args.image_path)

    if output_path:
        print_status(f"\nSegmentation mask saved to: {output_path}", "success")
    else:
        print_status("\nSegmentation cancelled.", "warning")


if __name__ == "__main__":
    main()
