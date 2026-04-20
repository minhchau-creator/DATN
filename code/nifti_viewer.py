"""
NIfTI Viewer - Interactive 2D viewer cho file NIfTI
Xem NIfTI files trực tiếp trong terminal/CLI
"""

import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
from pathlib import Path
import argparse


class NIfTIViewer:
    """Interactive NIfTI 3D viewer"""
    
    def __init__(self, nifti_path: str):
        """
        Khởi tạo viewer
        
        Args:
            nifti_path: Đường dẫn đến file .nii.gz
        """
        self.nifti_path = Path(nifti_path)
        
        if not self.nifti_path.exists():
            raise FileNotFoundError(f"File not found: {nifti_path}")
        
        # Load NIfTI file
        self.img = nib.load(self.nifti_path)
        self.data = self.img.get_fdata()
        
        print(f"✅ Loaded: {self.nifti_path.name}")
        print(f"   Shape (D, H, W): {self.data.shape}")
        print(f"   Data type: {self.data.dtype}")
        print(f"   Min: {self.data.min():.1f}, Max: {self.data.max():.1f}")
        print(f"   Mean: {self.data.mean():.1f}, Std: {self.data.std():.1f}")
        
        # Khởi tạo slices
        self.slice_idx = self.data.shape[0] // 2
        self.window_center = (self.data.min() + self.data.max()) / 2
        self.window_width = self.data.max() - self.data.min()
        
        self.create_figure()
    
    def create_figure(self):
        """Tạo interactive figure"""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(f"NIfTI Viewer: {self.nifti_path.name}", fontsize=14, fontweight='bold')
        
        # Main image plot
        self.ax_img = plt.subplot(2, 2, 1)
        self.im = self.ax_img.imshow(self.get_windowed_slice(), cmap='gray')
        self.ax_img.set_title("2D Slice (Axial)")
        plt.colorbar(self.im, ax=self.ax_img, label="HU")
        
        # Histogram
        self.ax_hist = plt.subplot(2, 2, 2)
        self.update_histogram()
        
        # Info text
        self.ax_info = plt.subplot(2, 2, (3, 4))
        self.ax_info.axis('off')
        self.update_info_text()
        
        # Slider cho slice
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.02])
        self.slider_slice = mwidgets.Slider(
            ax_slider, 'Slice', 0, self.data.shape[0]-1, 
            valinit=self.slice_idx, valstep=1, color='cyan'
        )
        self.slider_slice.on_changed(self.update_slice)
        
        # Slider cho window center
        ax_wc = plt.axes([0.2, 0.05, 0.6, 0.02])
        self.slider_wc = mwidgets.Slider(
            ax_wc, 'Window Center', self.data.min(), self.data.max(),
            valinit=self.window_center, color='yellow'
        )
        self.slider_wc.on_changed(self.update_window)
        
        # Slider cho window width
        ax_ww = plt.axes([0.2, 0.02, 0.6, 0.02])
        self.slider_ww = mwidgets.Slider(
            ax_ww, 'Window Width', 1, self.data.max() - self.data.min(),
            valinit=self.window_width, color='orange'
        )
        self.slider_ww.on_changed(self.update_window)
        
        # Keyboard shortcuts info
        info_text = (
            "⌨️  Keyboard Shortcuts:\n"
            "  UP/DOWN: Slice navigation\n"
            "  W: Reset window/level\n"
            "  R: Reset all\n"
            "  S: Save screenshot"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=9, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    
    def get_windowed_slice(self):
        """Áp dụng window/level lên slice hiện tại"""
        slice_data = self.data[self.slice_idx]
        
        # Window/Level (Hounsfield windowing)
        low = self.window_center - self.window_width / 2
        high = self.window_center + self.window_width / 2
        
        windowed = np.clip(slice_data, low, high)
        windowed = (windowed - low) / (high - low)
        
        return windowed
    
    def update_slice(self, val):
        """Update khi thay đổi slice"""
        self.slice_idx = int(self.slider_slice.val)
        self.im.set_data(self.get_windowed_slice())
        self.update_info_text()
        self.update_histogram()
        self.fig.canvas.draw_idle()
    
    def update_window(self, val):
        """Update khi thay đổi window/level"""
        self.window_center = self.slider_wc.val
        self.window_width = self.slider_ww.val
        self.im.set_data(self.get_windowed_slice())
        self.fig.canvas.draw_idle()
    
    def update_histogram(self):
        """Update histogram của slice hiện tại"""
        self.ax_hist.clear()
        slice_data = self.data[self.slice_idx]
        self.ax_hist.hist(slice_data.flatten(), bins=100, color='blue', alpha=0.7)
        self.ax_hist.set_title("Histogram")
        self.ax_hist.set_xlabel("HU")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.axvline(self.window_center, color='red', linestyle='--', label='Center')
        self.ax_hist.legend()
    
    def update_info_text(self):
        """Update thông tin hiển thị"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        slice_data = self.data[self.slice_idx]
        
        info = (
            f"📊 NIfTI Info: {self.nifti_path.name}\n"
            f"📐 Volume shape: {self.data.shape}\n"
            f"🔢 Data type: {self.data.dtype}\n"
            f"\n"
            f"📍 Current Slice: {self.slice_idx} / {self.data.shape[0]-1}\n"
            f"📈 Slice stats:\n"
            f"    Min: {slice_data.min():.1f} HU\n"
            f"    Max: {slice_data.max():.1f} HU\n"
            f"    Mean: {slice_data.mean():.1f} HU\n"
            f"    Std: {slice_data.std():.1f} HU\n"
            f"\n"
            f"🎛️  Window/Level:\n"
            f"    Center: {self.window_center:.1f} HU\n"
            f"    Width: {self.window_width:.1f} HU\n"
            f"    Range: [{self.window_center - self.window_width/2:.1f}, "
            f"{self.window_center + self.window_width/2:.1f}]"
        )
        
        self.ax_info.text(0.05, 0.95, info, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def on_key(self, event):
        """Xử lý keyboard events"""
        if event.key == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.data.shape[0] - 1)
            self.slider_slice.set_val(self.slice_idx)
        elif event.key == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
            self.slider_slice.set_val(self.slice_idx)
        elif event.key == 'w':
            # Reset window/level
            self.window_center = (self.data.min() + self.data.max()) / 2
            self.window_width = self.data.max() - self.data.min()
            self.slider_wc.set_val(self.window_center)
            self.slider_ww.set_val(self.window_width)
        elif event.key == 'r':
            # Reset all
            self.slice_idx = self.data.shape[0] // 2
            self.window_center = (self.data.min() + self.data.max()) / 2
            self.window_width = self.data.max() - self.data.min()
            self.slider_slice.set_val(self.slice_idx)
            self.slider_wc.set_val(self.window_center)
            self.slider_ww.set_val(self.window_width)
        elif event.key == 's':
            # Save screenshot
            output_path = self.nifti_path.parent / f"{self.nifti_path.stem}_slice{self.slice_idx}.png"
            self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Screenshot saved: {output_path}")
    
    def show(self):
        """Hiển thị viewer"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive NIfTI Viewer")
    parser.add_argument('nifti_path', type=str, help="Path to .nii.gz file")
    
    args = parser.parse_args()
    
    viewer = NIfTIViewer(args.nifti_path)
    viewer.show()


if __name__ == "__main__":
    main()
