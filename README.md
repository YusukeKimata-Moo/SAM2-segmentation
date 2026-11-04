# SAM2-segmentation

Use the Colab notebooks in this repository to segment time-lapse images with SAM 2.1 and fine-tune a model on your own annotated images.<br>
ref: Notebook in *Meta Research* repository (https://github.com/facebookresearch/sam2/blob/main/notebooks/).<br>

- [Segment time-lapse images](#segmentation)
- [Fine-tune a model on your own data](#fine-tuning)

## Segmentation

Open [SAM2-segmentation.ipynb](SAM2-segmentation.ipynb) to segment a video using pretrained or fine-tuned weights.

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

## Latest Update
### **Fine-tuning support (2026/11/02)**
Added [SAM2-finetuning.ipynb](SAM2-finetuning.ipynb) to fine-tune SAM 2.1 Base+ or Large on annotated image frames. The [segmentation notebook](SAM2-segmentation.ipynb) now supports loading fine-tuned weights.

### **Revision for flexible reference frame selection (2025/09/22)**
For initial segmentation, you can select not only the first frame but also any other frame as the reference.
   
## Citation
- SAM 2: Segment Anything in Images and Videos<br>
  Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer<br>
  _arXiv_: 2408.00714, (2024), https://arxiv.org/abs/2408.00714
- KymoTip: High-throughput Characterization of Tip-growth Dynamics in Plant Cells<br>
  Zichen Kang, Yusuke Kimata, Tomonobu Nonoyama, Toru Ikeuchi, Kazuyuki Kuchitsu, Satoru Tsugawa, Minako Ueda<br>
  _Plant Journal_ 125:e70691, (2026), [https://doi.org/10.1101/2025.06.27.661917](https://doi.org/10.1111/tpj.70691)

## Fine-tuning

Use [SAM2-finetuning.ipynb](SAM2-finetuning.ipynb) to adapt SAM 2.1 Base+ or Large to your images. You provide video frames and their segmentation masks on Google Drive. The notebook trains on individual annotated frames and lets you compare video tracking before and after fine-tuning.

You will need a Google Colab GPU runtime, space on Google Drive for your data and model weights, and the input files described below.

### 1. Prepare images and masks

| Item | Required format |
| --- | --- |
| Input images | One frame per JPEG file (`.jpg` or `.jpeg`), in grayscale or RGB. Extract frames from video files before using this notebook. |
| Segmentation masks | PNG files containing **0 for background** and **1 or 255 for the target**. Use a single foreground value throughout each mask. |
| Target | One foreground target per mask. Separate object IDs and colored annotation overlays are not supported. |
| Image–mask pairs | Use the same video folder name and filename stem: for example, `images/video_01/000.jpg` pairs with `masks/video_01/000.png`. Each pair must have identical pixel dimensions and alignment. |
| Image sizes | Different videos and training images may have different dimensions. For video tracking evaluation, all frames within the same video must share the same dimensions. |
| Frame names and order | Names do not have to be numeric or consecutive. Frames are read in natural filename order (`frame2` before `frame10`), so choose names that preserve their time order. Avoid two images with the same stem in one video folder. |

You do **not** need to annotate every frame. Frames with masks are used for image training and validation; unlabeled frames in test videos are still included in tracking. Masks without matching images are reported and ignored. A mask with no target can be included in video evaluation, but it is excluded from image training and validation because those steps need an initial point inside the target.

For tracking evaluation, provide a nonempty mask on an early frame and masks on the later frames you want to score. Tracking starts at the first annotated frame containing the target. That initial frame is excluded from the tracking score.

Save masks as binary PNGs. If an existing mask is stored as JPEG, threshold it to the background/foreground values above and save it as PNG. Simply changing the filename extension does not convert the image.

### 2. Organize the data on Google Drive

Create the following folders under **My Drive**. Use a different video ID, such as `video_01`, for each source video.

```text
Colab Notebooks/
└─ SAM2-segmentation/
   └─ finetune_data/
      ├─ images/
      │  ├─ video_01/
      │  │  ├─ 000.jpg
      │  │  ├─ 001.jpg
      │  │  └─ 002.jpg
      │  ├─ video_02/
      │  │  └─ ...
      │  └─ video_03/
      │     └─ ...
      └─ masks/
         ├─ video_01/
         │  ├─ 000.png
         │  └─ 002.png
         ├─ video_02/
         │  └─ ...
         └─ video_03/
            └─ ...
```

In this example, `001.jpg` is an unlabeled frame. The number of videos, frames, and annotated frames can vary.

By default, the notebook assigns **60% of the videos to training, 20% to validation, and 20% to testing**, rounded to whole videos. Frames from one video folder stay together. This three-way split requires at least three labeled video folders, and the training and validation groups must each contain nonempty masks. Use independent source videos or samples for evaluation; copies or transformed versions of one recording do not provide an independent test set.

### 3. Open the notebook and choose settings

Open [SAM2-finetuning.ipynb](SAM2-finetuning.ipynb) in Google Colab and select a **GPU** runtime. Run the cells from the top and allow access to Google Drive when prompted. The first setup downloads SAM2 and installs the required libraries. If Colab asks you to restart the runtime after installation, restart and begin again from **Setup**.

In **Prepare data**, review these settings before continuing:

| Setting | What to enter |
| --- | --- |
| `WORK_DIR` | The working folder on Drive. The default matches the layout above: `/content/drive/MyDrive/Colab Notebooks/SAM2-segmentation`. |
| `DATA_ROOT` | The folder containing `images/` and `masks/`. The default is `WORK_DIR / 'finetune_data'`. |
| `MODEL_SIZE` | `'base_plus'` (default) or `'large'`. This selects the model for both training and video evaluation. |
| `EPOCHS` | How many passes to make through the training images. The default is `20`; use `2` for a short test run. |
| `LEARNING_RATE` | How strongly the weights change during learning. The default is `5e-6`, a starting point to compare on your validation data. |
| `BATCH_SIZE` | Images processed in one training step. Start with the default, `1`. |
| `FREEZE_IMAGE_ENCODER` | Keep `True` to retain pretrained image feature extraction while adapting the segmentation. Setting it to `False` also trains that part of the model and increases GPU memory requirements. |
| `TRAIN_AUGMENT` | `True` by default. Set to `False` to train on original frames only. |
| `TRAIN_AUGMENTED_COPIES` | Extra augmented views per original frame in each epoch; default `1`. Set to `0` for originals only. |
| `TRAIN_AUGMENT_SCOPE` | `'frame'` (default) samples independently for each frame. `'video'` shares the operation and angle/factor across all frames of a video for each copy in an epoch. |
| `TRAIN_HFLIP` / `TRAIN_VFLIP` | Enable horizontal / vertical flips. Both default to `True`; set either to `False` if that orientation is unsuitable for your images. |
| `TRAIN_ROTATION_DEGREES` | Default `15`: sample a rotation angle uniformly from -15 to +15 degrees. Set to `0` to disable rotation. |
| `TRAIN_BRIGHTNESS_RANGE` | Default `None` disables brightness. Set two multiplicative factors, for example `(0.9, 1.1)`, to sample uniformly from slightly darker to slightly brighter. Brightness affects images only. |
| `SPLIT_RATIOS` / `SEED` | Video split proportions and the seed used to reproduce the split. Keep them unchanged when comparing model settings on the same dataset. |
| `CACHE_DATA` | Set to `True` to copy the input data to the Colab runtime for faster repeated reads. The initial copy can take time; the files on Drive are preserved. |

For example, to try Large with a short run:

```python
MODEL_SIZE = 'large'
EPOCHS = 2
BATCH_SIZE = 1
FREEZE_IMAGE_ENCODER = True
```

#### Training augmentation

With the defaults, each epoch includes every original training frame once and one extra augmented view. Each extra view selects **exactly one** enabled operation with equal probability; operations are not combined. With horizontal flip, vertical flip, and rotation enabled, each has a one-third chance. Enabling brightness as well gives each of the four a one-quarter chance.

Geometry is applied to the image and mask together before the existing 1024-square resize and normalization. Rotation uses bilinear image interpolation and nearest-neighbor mask interpolation, expands the canvas to retain the source field, and fills new pixels with black/background. This keeps the same sampled rotation for every frame in video scope without frame-specific fallback. Source files, validation images, and test videos stay unchanged.

Choose the settings in **Prepare data** before running the later cells. For frame-specific augmentation with brightness enabled:

```python
TRAIN_AUGMENT = True
TRAIN_AUGMENTED_COPIES = 1
TRAIN_AUGMENT_SCOPE = 'frame'
TRAIN_HFLIP = True
TRAIN_VFLIP = True
TRAIN_ROTATION_DEGREES = 15
TRAIN_BRIGHTNESS_RANGE = (0.9, 1.1)
```

For two extra views with consistent augmentation across each video's frames:

```python
TRAIN_AUGMENT = True
TRAIN_AUGMENTED_COPIES = 2
TRAIN_AUGMENT_SCOPE = 'video'
TRAIN_HFLIP = True
TRAIN_VFLIP = True
TRAIN_ROTATION_DEGREES = 15
TRAIN_BRIGHTNESS_RANGE = None
```

To disable all training augmentation, set:

```python
TRAIN_AUGMENT = False
```

Setting copies to `0`, or turning off both flips, rotation (`0`), and brightness (`None`), also leaves originals only. Draws are reproducible from `SEED`, epoch, frame path (frame scope) or video ID (video scope), and copy number; shuffle/access order does not affect them. Each new epoch draws again. Video scope controls transform consistency; training still processes individual frames without learning temporal memory.

The notebook prints the original training frame count and the samples per epoch. With augmentation active, `N` original frames and `C` copies produce `N * (1 + C)` samples. At `BATCH_SIZE = 1`, the default one extra copy doubles the steps per epoch; two copies triple them. All original and augmented views are shuffled together. More copies therefore increase training time per epoch. The augmentation settings are saved in `run_config.json` and checkpoint metadata.

### 4. Train and evaluate

Continue running the cells in order. Point prompts are generated from the supplied masks, so you do not need to click on images during this workflow.

| Notebook section | What happens |
| --- | --- |
| **Split videos and save the manifest** | Reads the image–mask pairs, reports the number of videos and labels in each group, and creates an output folder for this run. With caching enabled, it also copies the data to the runtime. |
| **Load model** | Downloads the selected pretrained weights on first use and loads the model. |
| **Train** | Trains for the requested number of epochs and validates after each epoch. The weights with the highest validation Dice are saved as `best.pt`. |
| **Training curves** | Displays training loss and validation Dice/IoU over time. |
| **Evaluate videos** | Compares the pretrained and fine-tuned weights on the test videos using the same initial point and no later correction clicks. It also saves the predicted masks as PNGs. |
| **Compare results** | Displays the tracking scores before and after fine-tuning. |
| **Compare segmentation masks** | Displays the original image and both predictions side by side, with red mask overlays and outlines. |

Validation measures segmentation on individual images, each with one initial foreground point and no correction clicks. Video evaluation measures tracking after the initial prompted frame. Dice and IoU range from 0 to 1; higher values indicate closer agreement with your supplied masks. Frame scores are averaged within each video, and then videos are averaged equally.

The notebook runs all requested epochs while keeping the best validation weights. `best.pt` may therefore come from an earlier epoch, rather than the final one. Tune settings using validation data and reserve the test videos for the final comparison.

To choose a different test video, run `print(list(test_sequences))` in a new cell to see the available video IDs. In **Compare segmentation masks**, set `VIDEO_TO_SHOW` to one of those IDs as a string, then run the display cell. For example:

```python
VIDEO_TO_SHOW = 'video_03'  # Replace with one of the available test video IDs.
FRAME_INDICES = None
```

With `FRAME_INDICES = None`, the display shows up to three frames spaced through the video after the initial prompt. To choose frames yourself, supply a list of **zero-based positions in the loaded video**: `0` is the first frame, regardless of its filename. For example, `[1, 5, 9]` selects the second, sixth, and tenth frames of a ten-frame video prompted at frame `0`. You can display the prompted frame or any later frame; frames before the prompt are outside the forward tracking result. Saved masks are reused when changing the display.

### 5. Find your results

Each run is saved separately on Drive. With the default settings, the location is:

```text
Colab Notebooks/SAM2-segmentation/finetune_runs/<run-folder>/
```

The notebook prints the folder as `Run directory:`. In a new Colab cell, run `print(RUN_DIR)` to display it again.

| File or folder | Contents |
| --- | --- |
| `best.pt` | The fine-tuned weights to use for segmentation. |
| `best_metadata.json` | The selected model size, best epoch, and validation scores. |
| `history.csv` / `training_curves.png` | Training loss and validation scores over the run. |
| `test_summary.csv` / `test_comparison.png` | Overall video tracking scores before and after fine-tuning. |
| `test_videos.csv` / `test_video_frames.csv` | Scores for each test video and each scored frame. |
| `video_masks/` | Original-size binary PNG predictions, organized by model and video. |
| `segmentation_comparison_<video_id>.png` | The side-by-side comparison for the selected video and frames. |

### 6. Use the fine-tuned weights for segmentation

At the end of the fine-tuning notebook, **Use the checkpoint** prints the model size and the full path to `best.pt`. Copy those values into the **Load model** settings in [SAM2-segmentation.ipynb](SAM2-segmentation.ipynb):

```python
MODEL_SIZE = 'large'  # Use the same size you selected for training.
CUSTOM_CHECKPOINT = '/content/drive/MyDrive/Colab Notebooks/SAM2-segmentation/finetune_runs/<run-folder>/best.pt'
```

Replace `<run-folder>` with your saved run folder, or paste the exact path printed by the notebook. Then follow the segmentation instructions above. The segmentation notebook uses numeric JPEG filenames such as `000.jpg`, `001.jpg`, and so on.

To try another training setting, change the configuration in **Prepare data** and rerun from that cell downward, including **Split videos and save the manifest**. A new run folder is created, allowing you to keep the previous results for comparison.
