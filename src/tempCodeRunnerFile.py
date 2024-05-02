import argparse
import logging
import tkinter as tk
from tkinter import *


import tkinter.font as font
import webbrowser
import random

# from readme_renderer import txt

from src.clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector
from src.face_embedding.faces_embedding import GenerateFaceEmbedding

from src.predictor.facePredictor import FacePredictor