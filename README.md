# yolo_detection
# This is faster object detection model as compared to maskrcnn with a bit compromise on the detection accuracy

# get started
git clone https://github.com/yangyin3027/yolo_detection.git

# Generate a conda environment
conda create -n [env name]
# Activate the environment
conda activate [env name]
# install dependencies
pip install -r requirements.txt

# Ready to run
# Command line 
python yolo_detector.py --img [img file path] --threshold 0.85 
