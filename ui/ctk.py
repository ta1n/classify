import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

class ImageViewer(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()  # Pack the frame within the main window
        self.create_widgets()

    def create_widgets(self):
        self.image_label = ctk.CTkLabel(self, text="No Image Selected", image=None)
        self.image_label.pack(padx=20, pady=20)

        self.open_button = ctk.CTkButton(self, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

    def open_image(self):
        self.filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.gif")]
        )

        if self.filename:
            self.display_image()

    def display_image(self):
        try:
            # Use PIL (Pillow) for image processing
            from PIL import Image, ImageTk

            image = Image.open(self.filename)

            # Resize image if necessary
            max_width, max_height = 800, 600  # Adjust these values as needed
            image.thumbnail((max_width, max_height), Image.ANTIALIAS)

            # Convert image to a format compatible with Tkinter
            image = ImageTk.PhotoImage(image)

            # Update the label with the new image
            self.image_label.configure(image=image, text="")
            self.image_label.image = image  # Keep a reference to avoid garbage collection

        except (FileNotFoundError, PermissionError) as e:
            self.image_label.configure(text="Error: Could not open image", image=None)
            print(f"Error opening image: {e}")

if __name__ == "__main__":
    root = ctk.CTk()
    root.geometry("500x400")
    root.title("Image Viewer")
    app = ImageViewer(root)
    root.mainloop()

