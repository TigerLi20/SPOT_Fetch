# LEGS_POMDP: LanguagE and Gesture Object Search

### Introduction

### Demo

### :rocket: Quick Start

#### Setting Up the Conda Environment

   ```bash
   conda env create -n LEGS 
   conda activate LEGS
   pip install -r requirements.txt
   ```

#### Setting Up SoM(Set-of-Mark Visual Prompting for GPT-4V)
This project includes the Set-of-Mark (SoM) repository as a submodule for visual grounding tasks. Follow these steps to set up and use SoM:

1. Clone the Repository with Submodules

```
git clone --recurse-submodules git@github.com:h2r/LEGS-POMDP.git
```

If you haven’t cloned this repository yet, use the --recurse-submodules flag to ensure the submodule is cloned as well:
```
git submodule update --init --recursive
```
2. Install dependencies and download the pretrained models
```
cd object_detection/SoM
# install Deformable Convolution for Semantic-SAM
cd ops && bash make.sh && cd ..
# download pretrained models
sh download_ckpt.sh
```

### System Implementation

#### VLM

**Step 1(Part 1): receive audio**

Speech Recognition/speech_reg.py

Use whisper to transcribe and save to a hidden temp txt. File in the folder. 

Input source: Voice 

Output: 

- .tmp/temp_audio.wav
- .tmp/.transcription.txt

**Step 1(Part 2): Capture image (on Spot)**

[TODO]

Input source: Spot camera

Output: /.tmp/image.png

**Step 2: Segment image**
SoM-Segmentation.py

Summary: Segmented image with marks on the image

Input: ./.tmp/image.png

Output: 

- Json: ./.tmp/detection_confidence.json(mark_id, bounding box, predicted_iou_score)
- Png: /.tmp/annotated_image.png, /.tmp/mask.png (red channel corresponds to the mask id)


**Step 3: Extract object from language**

SoM_GPT4.py

Summary: process the segmented image and transcription to select the likely objects for the fetching task. 

Input: image[/.tmp/annotated_image.png], transcription[.tmp/.transcription.txt]

Return: add "lang_prob"(0 or 1) to ./.tmp/detection_confidence.json


**Step 4: Combine probability**

Combined_probability.py

Summary: process the raw probability to generate submodule and combined posterior probabilities 

Input: json of language and gesture probability

Output [./.tmp/detection_confidence.json]:

- Language submodule probability: "lang_prob_normalized"

- Gesture Submodule probability: TODO

- Combined probability: TODO



