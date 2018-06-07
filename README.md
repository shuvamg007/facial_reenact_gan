# Facial Re-enactment using pix2pix
This is an ongoing effort to re-enact faces using AdversarialNets. Here, we are utilizing pix2pix to generate the patches for mouth. The initial demo was performed on the videos of former president of the United States, Barack Obama. The initial version makes uses of two videos, namely a source and a target to morph the face.
## Pre-requisites
- Python 3
- TensorFlow 1.2 (specifically)
- OpenCV >= 3.0
- Dlib 19.4
## Instructions

##### 1. Clone this repo
```
git clone git@github.com:shuvamg007/facial_reenact_gan.git
```
##### 2. Generate Training Data
While training data can be generated using the instructions given in pix2pix, the following command reduces much of the hassle.
```
python3 pre_process.py --skip 3 --predictor /path/to/shape_predictor_68_face_landmarks.dat --folder /path/to/videos/
```

Input:

- `folder` - folder containing the video files using which the data set is to be created.
- `skip` - number of frames to be skipped before using the next frame (default=1).
- `predictor` - facial landmark model to be used for landmark detection. A pre-trained facial landmark model is available [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Output:

- A folder `proc_image` containing merged images will be created.

##### 3. Train Model

Clone the repo from Christopher Hesse's pix2pix TensorFlow implementation
```
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
```

Go into the pix2pix-tensorflow folder
```
cd pix2pix-tensorflow/
```

Reset to Tensolrflow 1.2 version
```
git reset --hard d6f8e4ce00a1fd7a96a72ed17366bfcb207882c7
```

Move the processed image folder into the pix2pix-tensorflow folder
```
mv ../proc_image photos
```

Split into train and val set
```
python tools/split.py --dir photos/proc_image
```  

Train the model on the data
```
python pix2pix.py \
  --mode train \
  --output_dir face_reenact_model \
  --max_epochs 200 \
  --input_dir photos/combined/train \
  --which_direction AtoB
```

For more information about how to train your own model, take a look at Christopher Hesse's [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) implementation.

##### 4. Export Model

First, we need to reduce the trained model :
```
python reduce_model.py --ip_folder face_reenact_model --op_folder face_reenact_reduced
```
Input:
- `ip_folder` - model folder to be imported.
- `op_folder` - folder where reduced model is to be saved.

Output:
- A reduced model with smaller weight file.

Then, we freeze the reduced model to a single file.
```
python freeze_model.py --folder face_reenact_reduced
```

Input:

- `folder` - folder containing the reduced model.

Output:

- A model file, `final_model.pb`, in the model folder.

##### 5. Reenacting using trained model

We first process the source and target videos
```
python3 morph_op.py --source /path/to/source --target /path/to/target --predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Input:

- `source` - source video file.
- `target` - target video file.
- `predictor` - facial landmark model to be used for landmark detection.

Output:

- Three folders `align_src`, `align_tgt` & `frames` containing aligned frames and images will be created.

We will then generate patches with the appropriate mouth shapes
```
python3 generate_tps_fixed.py --model /path/to/final_model.pb --predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Input:

- `model` - path to the frozen model file.
- `predictor` - facial landmark model to be used for landmark detection.

Output:

- A folder `tgt_oriented` containing patched images will be created.

Lastly, we need to blend the patch onto the final frame
```
python3 process_vid.py --predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Input:

- `predictor` - facial landmark model to be used for landmark detection.

Output:

- A folder `final_op` containing final frames will be created.

##### 6. Creating video

After the generation of frames, one can easily merge the output in `final_op` using FFMPEG to stich the video.

## Acknowledgments
Kudos to [Christopher Hesse](https://github.com/christopherhesse) for his amazing pix2pix TensorFlow implementation and [Dat Tran](https://github.com/datitran) for his awesom work on face2face. 
