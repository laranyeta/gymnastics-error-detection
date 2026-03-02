#### TESTING SAMPLE (WIP)
## first samples and test will be shown through this code
## via balance beam videos
from utils.data import read_video, filter_annotations
from utils.visualization import *

filter_annotations("dataset/original/finegym_v1.1.json")
read_video("dataset/video_test/front_aerial.mp4")