import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VirtualOpticsLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Optics Lab - MFAD Mini Project")
        self.geometry("1400x800")
        self.configure(bg="#f4f4f4")
        
        # State Variables
        self.raw_image = None
        self.original_img_matrix = None
        self.transformed_img_matrix = None
        
        self.u = 100.0
        self.f = 50.0
        self.v = 0.0
        self.m = 1.0
        self.R = 100.0
        self.screen_dist = 100.0
        
        self.nature = "Real"
        self.orientation = "Inverted"
        self.size_desc = "Same size"
        self.screen_req = True
        self.focus_error = 0.0
        
        # ABCD Matrices
        self.P_obj = np.eye(2)
        self.E_mat = np.eye(2)
        self.P_img = np.eye(2)
        self.System_mat = np.eye(2)
        self.A_mat = np.eye(3)

        self.last_canvas_w = 400
        self.last_canvas_h = 350

        self.setup_default_image()
        self.create_gui()
        
        # Delay the first simulation update slightly to let Tkinter calculate accurate canvas sizes
        self.after(200, self.update_simulation)

    def setup_default_image(self):
        """Create a default asymmetric 'F' image matrix to clearly show inversions."""
        img = np.ones((400, 400), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 40), (160, 360), 0, -1)    # Vertical stem
        cv2.rectangle(img, (160, 40), (300, 100), 0, -1)    # Top bar
        cv2.rectangle(img, (160, 160), (260, 220), 0, -1)   # Middle bar
        self.raw_image = img

    def create_gui(self):
        # MAIN LAYOUT
        self.sidebar = tk.Frame(self, width=300, bg="#2c3e50", padx=20, pady=20)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.main_area = tk.Frame(self, bg="#ecf0f1")
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # SIDEBAR CONTROLS
        tk.Label(self.sidebar, text="Virtual Optics Lab", font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50").pack(pady=(0, 5))
        tk.Label(self.sidebar, text="MFAD Mini Project", font=("Helvetica", 12), fg="#1abc9c", bg="#2c3e50").pack(pady=(0, 20))
        
        tk.Label(self.sidebar, text="Optical Element:", fg="white", bg="#2c3e50").pack(anchor="w")
        self.element_var = tk.StringVar(value="Convex Lens")
        self.element_cb = ttk.Combobox(self.sidebar, textvariable=self.element_var, state="readonly")
        self.element_cb['values'] = ("Convex Lens", "Concave Lens", "Concave Mirror", "Convex Mirror")
        self.element_cb.pack(fill=tk.X, pady=(0, 15))
        self.element_cb.bind("<<ComboboxSelected>>", self.on_input_change)

        self.u_slider = self.create_slider("Object Distance (u)", 10, 300, 100)
        self.f_slider = self.create_slider("Focal Length / Radius", 10, 300, 50)
        self.screen_slider = self.create_slider("Screen Distance", 10, 500, 100)
        
        tk.Button(self.sidebar, text="Select Input Image", command=self.load_image, bg="#3498db", fg="white").pack(fill=tk.X, pady=10)
        tk.Button(self.sidebar, text="Show Linear Algebra", command=self.show_linear_algebra, bg="#9b59b6", fg="white").pack(fill=tk.X, pady=10)
        
        # RESULTS PANEL IN SIDEBAR
        self.result_frame = tk.Frame(self.sidebar, bg="#34495e", padx=10, pady=10)
        self.result_frame.pack(fill=tk.X, pady=20)
        self.result_text = tk.StringVar()
        tk.Label(self.result_frame, textvariable=self.result_text, justify=tk.LEFT, fg="white", bg="#34495e", font=("Consolas", 10)).pack(anchor="w")
        
        tk.Button(self.sidebar, text="Exit", command=self.quit, bg="#e74c3c", fg="white").pack(side=tk.BOTTOM, fill=tk.X)

        # MAIN AREA (Top: Images, Bottom: Optical Bench)
        self.images_frame = tk.Frame(self.main_area, bg="#ecf0f1", height=350)
        self.images_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Center Canvases
        self.orig_frame = tk.LabelFrame(self.images_frame, text="Original Input Matrix (Fills Entire Panel)", bg="white")
        self.orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.orig_canvas = tk.Canvas(self.orig_frame, bg="#2c3e50", highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        self.formed_frame = tk.LabelFrame(self.images_frame, text="Formed Image (Mathematically Zoomed & Centered)", bg="white")
        self.formed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.formed_canvas = tk.Canvas(self.formed_frame, bg="#2c3e50", highlightthickness=0)
        self.formed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Handle Window Resizing to adjust image scale automatically
        self.orig_canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Matplotlib Bench
        self.bench_frame = tk.LabelFrame(self.main_area, text="Optical Bench Ray Diagram", bg="white")
        self.bench_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.figure, self.ax = plt.subplots(figsize=(8, 3))
        self.figure.patch.set_facecolor('#ffffff')
        self.canvas_matplot = FigureCanvasTkAgg(self.figure, master=self.bench_frame)
        self.canvas_matplot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_slider(self, label, min_val, max_val, default):
        frame = tk.Frame(self.sidebar, bg="#2c3e50")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label, fg="white", bg="#2c3e50").pack(anchor="w")
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, bg="#2c3e50", fg="white", highlightthickness=0, command=lambda _: self.on_input_change())
        slider.set(default)
        slider.pack(fill=tk.X)
        return slider

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.raw_image = img
                self.update_simulation()
            else:
                messagebox.showerror("Error", "Could not load image.")

    def on_canvas_resize(self, event):
        """Re-scales the base matrix if the user resizes the application window."""
        if abs(event.width - self.last_canvas_w) > 10 or abs(event.height - self.last_canvas_h) > 10:
            self.last_canvas_w = max(event.width, 400)
            self.last_canvas_h = max(event.height, 350)
            self.update_simulation()

    def on_input_change(self, event=None):
        self.update_simulation()

    def calculate_physics(self):
        self.u = self.u_slider.get()
        val = self.f_slider.get()
        self.screen_dist = self.screen_slider.get()
        elem = self.element_var.get()
        
        if "Mirror" in elem:
            self.R = val
            self.f = self.R / 2.0
            is_mirror = True
        else:
            self.f = val
            is_mirror = False
            
        if "Concave Lens" in elem or "Convex Mirror" in elem:
            self.f = -self.f 
            
        if self.u == self.f:
            self.v = float('inf')
            self.m = float('inf')
            self.nature = "At Infinity"
            self.orientation = "Undefined"
            self.size_desc = "Extremely large"
            self.screen_req = False
        else:
            self.v = (self.f * self.u) / (self.u - self.f)
            self.m = -self.v / self.u
            
            if self.v > 0:
                self.nature = "Real"
                self.orientation = "Inverted" if self.m < 0 else "Upright"
                self.screen_req = True
            else:
                self.nature = "Virtual"
                self.orientation = "Upright"
                self.screen_req = False
                
            abs_m = abs(self.m)
            if abs_m > 1.05:
                self.size_desc = "Magnified"
            elif abs_m < 0.95:
                self.size_desc = "Diminished"
            else:
                self.size_desc = "Same size"

        # 2. Matrix Multiplication Model
        self.P_obj = np.array([[1, self.u], [0, 1]])
        
        if is_mirror:
            self.E_mat = np.array([[1, 0], [-2/self.R if "Concave" in elem else 2/self.R, 1]])
        else:
            self.E_mat = np.array([[1, 0], [-1/self.f, 1]])
            
        if self.v == float('inf'):
            self.P_img = np.array([[1, 9999], [0, 1]])
        else:
            self.P_img = np.array([[1, self.v], [0, 1]])
            
        self.System_mat = self.P_img @ self.E_mat @ self.P_obj

    def process_image(self):
        cw = max(self.orig_canvas.winfo_width(), 400)
        ch = max(self.orig_canvas.winfo_height(), 350)

        # 1. Prepare Base Matrix to EXACTLY cover the panel area (Removes top/bottom padding)
        self.original_img_matrix = cv2.resize(self.raw_image, (cw, ch))
        
        img = self.original_img_matrix.copy()
        
        # 2. ZOOM IN / ZOOM OUT via Affine Transformation Matrix
        if self.m == float('inf'):
            scale = 5.0 # Max visual bound to prevent memory crash
        else:
            scale = abs(self.m)
            scale = min(max(scale, 0.1), 5.0) 

        new_trans_w, new_trans_h = int(cw * scale), int(ch * scale)
        
        self.A_mat = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        affine_warp_mat = self.A_mat[:2, :] 
        transformed = cv2.warpAffine(img, affine_warp_mat, (new_trans_w, new_trans_h))
        
        # 3. INVERSION via Matrix Flip
        if self.m != float('inf') and self.m < 0:
            transformed = cv2.flip(transformed, 0)
            
        # 4. BLURRING via Gaussian Convolution
        if self.screen_req:
            self.focus_error = abs(self.v - self.screen_dist)
        else:
            self.focus_error = 0.0 
            
        if self.focus_error > 0.5:
            sigma = self.focus_error * 0.1
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0: ksize += 1
            transformed = cv2.GaussianBlur(transformed, (ksize, ksize), sigma)
            
        self.transformed_img_matrix = transformed

    def display_images(self):
        """Render the exact processed matrices to the screen."""
        cw = max(self.orig_canvas.winfo_width(), 400)
        ch = max(self.orig_canvas.winfo_height(), 350)
        
        # Original 
        orig_pil = Image.fromarray(self.original_img_matrix)
        self.orig_img_tk = ImageTk.PhotoImage(orig_pil)
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.orig_img_tk)
        
        # Formed - This acts as a camera viewport. If magnified, it naturally crops at the edges.
        formed_pil = Image.fromarray(self.transformed_img_matrix)
        self.formed_img_tk = ImageTk.PhotoImage(formed_pil)
        self.formed_canvas.delete("all")
        self.formed_canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.formed_img_tk)

    def update_result_text(self):
        arrow = "↑" if self.orientation == "Upright" else "↓"
        if self.nature == "At Infinity":
            arrow = "∞"
            
        res = f"IMAGE FORMATION\n"
        res += "-"*20 + "\n"
        res += f"{self.nature} Image\n"
        res += f"{arrow} {self.orientation}\n"
        res += f"{self.size_desc}\n"
        
        if self.m != float('inf'):
            res += f"Magnification: {abs(self.m):.2f}×\n"
            res += f"Image Dist: {self.v:.1f} cm\n"
        else:
            res += "Magnification: ∞\nImage Dist: ∞\n"
            
        res += f"\nObserver Required: {'NO' if self.screen_req else 'YES'}\n"
        res += f"Screen Required: {'YES' if self.screen_req else 'NO'}\n"
        
        if self.screen_req:
            res += f"Focus Error: {self.focus_error:.1f} units\n"
            
        self.result_text.set(res)

    def update_bench(self):
        self.ax.clear()
        self.ax.axhline(0, color='black', linewidth=1)
        
        elem = self.element_var.get()
        is_mirror = "Mirror" in elem
        
        # True Matplotlib shapes for curved lenses and mirrors instead of straight lines
        y_vals = np.linspace(-15, 15, 100)
        
        if "Convex Lens" in elem:
            x_vals = 3 * np.cos(np.arcsin(y_vals/15)) 
            self.ax.plot(-x_vals, y_vals, color='blue', linewidth=2, label=elem)
            self.ax.plot(x_vals, y_vals, color='blue', linewidth=2)
        elif "Concave Lens" in elem:
            x_vals = 3 * (1 - np.cos(np.arcsin(y_vals/15))) + 1
            self.ax.plot(-x_vals, y_vals, color='blue', linewidth=2, label=elem)
            self.ax.plot(x_vals, y_vals, color='blue', linewidth=2)
            self.ax.plot([-x_vals[0], x_vals[0]], [15, 15], color='blue', linewidth=2)
            self.ax.plot([-x_vals[-1], x_vals[-1]], [-15, -15], color='blue', linewidth=2)
        elif "Concave Mirror" in elem:
            x_vals = -2 * (1 - np.cos(np.arcsin(y_vals/15)))
            self.ax.plot(x_vals, y_vals, color='blue', linewidth=3, label=elem)
            for y_mark in range(-14, 15, 3):
                idx = np.argmin(np.abs(y_vals - y_mark))
                self.ax.plot([x_vals[idx], x_vals[idx] + 3], [y_mark, y_mark - 2], color='gray', linewidth=1)
        elif "Convex Mirror" in elem:
            x_vals = 2 * (1 - np.cos(np.arcsin(y_vals/15)))
            self.ax.plot(x_vals, y_vals, color='blue', linewidth=3, label=elem)
            for y_mark in range(-14, 15, 3):
                idx = np.argmin(np.abs(y_vals - y_mark))
                self.ax.plot([x_vals[idx], x_vals[idx] - 3], [y_mark, y_mark - 2], color='gray', linewidth=1)
            
        self.ax.arrow(-self.u, 0, 0, 10, head_width=3, head_length=2, fc='red', ec='red', linewidth=2, length_includes_head=True)
        self.ax.text(-self.u, 12, 'Object', color='red', ha='center')

        if self.m != float('inf'):
            img_h = 10 * self.m
            img_x = -self.v if is_mirror else self.v
            self.ax.arrow(img_x, 0, 0, img_h, head_width=3, head_length=2, fc='green', ec='green', linewidth=2, length_includes_head=True)
            self.ax.text(img_x, img_h + (3 if img_h>0 else -5), 'Image', color='green', ha='center')
        
        if self.screen_req:
            sx = -self.screen_dist if is_mirror else self.screen_dist
            self.ax.axvline(sx, color='orange', linestyle='--', linewidth=2, label='Screen')
        else:
            eye_x = 50 if is_mirror else max(self.u, abs(self.v)) + 20
            self.ax.plot(eye_x, 0, marker=r'$\odot$', markersize=15, color='purple')
            self.ax.text(eye_x, 5, 'Eye', color='purple', ha='center')

        max_dist = max(self.u, abs(self.v) if self.v != float('inf') else 200, self.screen_dist)
        self.ax.set_xlim(-max_dist - 50, max_dist + 50)
        self.ax.set_ylim(-25, 25)
        self.ax.set_title("Optical Bench Ray Diagram", fontsize=10)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_yticks([])
        
        self.canvas_matplot.draw()

    def update_simulation(self):
        self.calculate_physics()
        if self.raw_image is not None:
            self.process_image()
            self.display_images()
        self.update_result_text()
        self.update_bench()

    def show_linear_algebra(self):
        win = tk.Toplevel(self)
        win.title("Linear Algebra Engine")
        win.geometry("600x600")
        win.configure(bg="#1e272e")
        
        text = tk.Text(win, bg="#1e272e", fg="#00d8d6", font=("Consolas", 12), padx=20, pady=20)
        text.pack(fill=tk.BOTH, expand=True)
        
        content = "=== VIRTUAL OPTICS LAB: MATRIX ENGINE ===\n\n"
        content += f"Base Panel Matrix Shape: {self.original_img_matrix.shape}\n"
        content += f"Matrix Data Type       : {self.original_img_matrix.dtype}\n\n"
        
        content += "1. Object Propagation Matrix P(u):\n"
        content += f"{np.array_str(self.P_obj, precision=2, suppress_small=True)}\n\n"
        
        content += f"2. Optical Element Matrix E (f={self.f:.1f}):\n"
        content += f"{np.array_str(self.E_mat, precision=4, suppress_small=True)}\n\n"
        
        content += "3. Image Propagation Matrix P(v):\n"
        content += f"{np.array_str(self.P_img, precision=2, suppress_small=True)}\n\n"
        
        content += "4. System Matrix = P(v) @ E @ P(u):\n"
        content += f"{np.array_str(self.System_mat, precision=2, suppress_small=True)}\n\n"
        
        content += "5. Affine Transformation Matrix (Scaling & Zoom):\n"
        content += f"{np.array_str(self.A_mat, precision=2, suppress_small=True)}\n\n"

        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = VirtualOpticsLab()
    app.mainloop()