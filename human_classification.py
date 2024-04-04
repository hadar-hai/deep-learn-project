import tkinter as tk
from PIL import ImageTk, Image
import os
import datetime
import random

class ImageChooserApp:
    def __init__(self, master, images):
        self.master = master
        self.images = images
        self.current_image_index = 0
        self.correct_guesses = 0
        
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.african_button = tk.Button(master, text="African", command=lambda: self.choose_category("African"))
        self.african_button.pack(side=tk.LEFT)
        
        self.asian_button = tk.Button(master, text="Asian", command=lambda: self.choose_category("Asian"))
        self.asian_button.pack(side=tk.RIGHT)
        
        self.quit_button = tk.Button(master, text="Quit Game", command=lambda: self.choose_category("Quit"))
        self.quit_button.pack(side=tk.TOP)
        
        self.correct_label = tk.Label(master, text=f"Correct Guesses: {self.correct_guesses}")
        self.correct_label.pack()

        self.show_image()

    def show_image(self):
        img = Image.open(self.images[self.current_image_index])
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference
       
    def choose_category(self, category):
        print(f"User chose category: {category}")
        user_choice = category
        true_category = os.path.basename(os.path.dirname(self.images[self.current_image_index]))
        correct = user_choice == true_category
        self.current_image_index += 1
        if correct:
            self.correct_guesses += 1
        self.correct_label.config(text=f"Correct Guesses: {self.correct_guesses}/{self.current_image_index}")
        # quit the app if we have shown all images or if the user choose to quit
        if self.current_image_index >= len(self.images) or category == "Quit":
            self.master.quit()
            return
        self.show_image()
        
    def get_correct_guesses(self):
        return self.correct_guesses
    
    def get_total_images_shown(self):
        return self.current_image_index

if __name__ == "__main__":
    root_tk = tk.Tk()
    root_tk.title("Image Chooser")

    images_dir = r"C:\Users\Data_Science\Documents\GitHub\deep-learn-project\input\dataset\test" # change this to the path of the images
    logs_dir = r".\logs"

    images = []
    for images_dir, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(images_dir, file))
    # randomly shuffle the images
    random.seed(0)
    random.shuffle(images)
                    
    app = ImageChooserApp(root_tk, images)
    root_tk.mainloop()
    correct_guesses = app.get_correct_guesses()
    images_shown = app.get_total_images_shown()
    
    # Human classification accuracy
    accuracy = (correct_guesses / images_shown)*100
    print(f"Human Accuracy: {accuracy:.2f}%")
    
    # write log file
    # time for file name
    time_ = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_title = f"human_classification_log_{time_}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_path = os.path.join(logs_dir, log_title + ".txt")
    with open(log_path, "w") as f:
        f.write(f"Correct Guesses: {correct_guesses}/{images_shown}\n")
        f.write(f"Human Accuracy: {accuracy:.2f}%\n")


