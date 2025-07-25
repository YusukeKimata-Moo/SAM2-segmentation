# SAM2-segmentation

This repository offers a Colab Notebook for segmentation of time-lapse image with SAM2 model.<br>
ref: Notebook in *Meta Research* repository (https://github.com/facebookresearch/sam2/blob/main/notebooks/).<br>

## Instructions

1. Create a directory named `Colab Notebooks/SAM2-segmentation` directly under `/content/drive/MyDrive/` on Google Drive.<br>

2. Place the JPEG files, each corresponding to a separate frame, into the input directory, located under `SAM2-segmentation/`.<br>
   **Note**: Only JPEG files with numeric filenames can be processed as input (e.g., `000.jpg`, `001.jpg`, ...).<br>
   On the ImageJ (Fiji) platform, you can prepare JPEG files that meet the requirements using the macro `jpg_for_SAM2.ijm`. <br>
   
3. Open the Notebook, and set the runtime type to GPU.<br>
   Then, run each cell sequentially (specify the directory names for input and output in the corresponding cell).<br>
   
   You can choose two types of output.<br>
      i. Mask image in grayscale<br>
      ii. Overlayed Masks on Original Images<br>

![Example](images/Example.gif)<br>
*Example of segmentation by SAM2 (Arabidopsis zygote)*
   
