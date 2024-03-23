import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pathlib
from PIL import Image


base_dir = "./dataset/"
train_dir = "./dataset/train/"
test_dir = "./dataset/test/"


# Walk through directory and list number of files
for dirpath, dirnames, filenames in os.walk(base_dir):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


data_dir = pathlib.Path(train_dir) # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
class_names = class_names.tolist() # convert numpy array to list
n_classes = len(class_names)
print(class_names, n_classes)


dir_dict = {train_dir: "Training", test_dir: "Testing"}

data = []

for type_dir, type_label in dir_dict.items():
    for class_name in class_names:
        target_folder = os.path.join(type_dir, class_name)
        for image_file in os.listdir(target_folder):
            img_path = os.path.join(target_folder, image_file)
            with Image.open(img_path) as img:
                data.append({
                    'class_name': class_name,
                    'type': type_label,
                    'img_path': img_path,
                    'shapes': img.size + (len(img.getbands()),)
                })

df = pd.DataFrame(data)
df.head()
df.type.value_counts()
df.class_name.value_counts()

# Create separate dataframes for Training and Testing
df_train = df[df['type'] == 'Training']
df_test = df[df['type'] == 'Testing']
df_test.head()
df_train.class_name.value_counts()
df_test.class_name.value_counts()


# Create a pie chart of class frequencies
plt.figure(figsize=(5, 4))
df['class_name'].value_counts().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 14})
plt.title('Class Frequencies', fontsize=16)
plt.ylabel('')
plt.show()
print()

plt.figure(figsize=(5, 4))
df['class_name'].value_counts().plot(kind='bar')
plt.title('Class Frequencies', fontsize=16)
plt.xlabel('Class Name', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()


# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.img_path[random_index[i]]))
    ax.set_title(df.class_name[random_index[i]])
plt.tight_layout()
plt.show()


# Here we create a list to store the random indices for each class
indices = []

# We iterate over each unique class in the DataFrame
for class_name in df['class_name'].unique():
    class_indices = df[df['class_name'] == class_name].index
    if len(class_indices) >= 5:
        # Select random indices from this class
        random_indices = np.random.choice(class_indices, 5, replace=False)
    else:
        # If the class has less than 5 instances, select all of them
        random_indices = class_indices
    indices.extend(random_indices)

# Plot the selected random images from each class
fig, axes = plt.subplots(nrows=len(indices)//5, ncols=5, figsize=(10, len(indices)//5*2),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.img_path[indices[i]]))
    ax.set_title(df.class_name[indices[i]])
plt.tight_layout()
plt.show()


