

import tensorflow as tf
# print(tf.test.is_gpu_available())



# import sys;
# print(f"OUTPUT{sys.path}")


# CODE ADDED FOR FLASK

from flask import Flask, request
from flask_cors import CORS
import base64
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  # This will enable CORS for all routes

cors = CORS(app, resources={r"/takeImages": {"origins": "*"}})






import argparse
import logging
import tkinter as tk
from tkinter import *


import tkinter.font as font
import webbrowser
import random

from readme_renderer import txt

# from clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
# from collect_trainingdata.get_faces_from_camera import TrainingDataCollector
# from face_embedding.faces_embedding import GenerateFaceEmbedding

# from predictor.facePredictor import FacePredictor
# from training.train_softmax import TrainFaceRecogModel



from src.clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector
from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector1
from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector2

from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector3
from src.collect_trainingdata.get_faces_from_camera import TrainingDataCollector4

from src.face_embedding.faces_embedding import GenerateFaceEmbedding

from src.predictor.facePredictor import FacePredictor
from src.training.train_softmax import TrainFaceRecogModel






# class RegistrationModule:
#     def __init__(self, logFileName):
#
#         self.logFileName = logFileName
#         self.window = tk.Tk()
#         # helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
#         self.window.title("Face Recognition")
#
#         # this removes the maximize button
#         self.window.resizable(0, 0)
#         window_height = 600
#         window_width = 880
#
#         screen_width = self.window.winfo_screenwidth()
#         screen_height = self.window.winfo_screenheight()
#
#         x_cordinate = int((screen_width / 2) - (window_width / 2))
#         y_cordinate = int((screen_height / 2) - (window_height / 2))
#
#         self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
#         # window.geometry('880x600')
#         self.window.configure(background='#ffffff')
#
#         # window.attributes('-fullscreen', True)
#
#         self.window.grid_rowconfigure(0, weight=1)
#         self.window.grid_columnconfigure(0, weight=1)
#
#         header = tk.Label(self.window, text="Face Recognition", width=80, height=2, fg="white", bg="#363e75",
#                           font=('times', 18, 'bold', 'underline'))
#         header.place(x=0, y=0)
#         clientID = tk.Label(self.window, text="ID", width=10, height=2, fg="white", bg="#363e75", font=('times', 15))
#         clientID.place(x=80, y=80)
#
#         displayVariable = StringVar()
#         self.clientIDTxt = tk.Entry(self.window, width=20, text=displayVariable, bg="white", fg="black",
#                                font=('times', 15, 'bold'))
#         self.clientIDTxt.place(x=205, y=80)
#
#         empID = tk.Label(self.window, text="Roll No", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
#         empID.place(x=450, y=80)
#
#         self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
#         self.empIDTxt.place(x=575, y=80)
#
#         empName = tk.Label(self.window, text="Name", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
#         empName.place(x=80, y=140)
#
#         self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
#         self.empNameTxt.place(x=205, y=140)
#
#         emailId = tk.Label(self.window, text="Email ID :", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
#         emailId.place(x=450, y=140)
#
#         self.emailIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
#         self.emailIDTxt.place(x=575, y=140)
#
#         mobileNo = tk.Label(self.window, text="Mobile No :", width=10, fg="white", bg="#363e75", height=2,
#                             font=('times', 15))
#         mobileNo.place(x=450, y=140)
#
#         self.mobileNoTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
#         self.mobileNoTxt.place(x=575, y=140)
#
#         lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
#                         font=('times', 15))
#         self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
#                                 activebackground="#e47911", font=('times', 15))
#         self.message.place(x=220, y=220)
#         lbl3.place(x=80, y=260)
#
#         self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
#                            font=('times', 15))
#         self.message.place(x=205, y=260)
#
#         # lbl3 = tk.Label(self.window, text="Attendance : ", width=15, fg="white", bg="#363e75", height=2,
#         #                 font=('times', 15))
#         # lbl3.place(x=80, y=440)
#         #
#         # self.message2 = tk.Label(self.window, text="", fg="#e47911", bg="#bbc7d4", activeforeground="#f8f9fa", width=52, height=2,
#         #                     font=('times', 15))
#         # self.message2.place(x=250, y=440)
#
#         takeImg = tk.Button(self.window, text="Take Images", command=self.collectUserImageForRegistration, fg="white", bg="#363e75", width=15,
#                             height=2,
#                             activebackground="#118ce1", font=('times', 15, ' bold '))
#         takeImg.place(x=80, y=350)
#
#         trainImg = tk.Button(self.window, text="Train Images", command=self.trainModel, fg="white", bg="#363e75", width=15,
#                              height=2,
#                              activebackground="#118ce1", font=('times', 15, ' bold '))
#         trainImg.place(x=350, y=350)
#
#         predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
#                              width=15,
#                              height=2,
#                              activebackground="#118ce1", font=('times', 15, ' bold '))
#         predictImg.place(x=600, y=350)
#
#         quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
#                                activebackground="#118ce1", font=('times', 15, 'bold'))
#         quitWindow.place(x=650, y=510)
#
#         link2 = tk.Label(self.window, text="Copyright, Simple App", fg="blue", )
#         link2.place(x=690, y=580)
#         # link2.pack()
#         link2.bind("<Button-1>", lambda e: self.callback("#"))
#         label = tk.Label(self.window)
#
#         self.window.mainloop()
#
#         logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
#                             level=logging.INFO,
#                             datefmt='%Y-%m-%d %H:%M:%S')
#
#     def getRandomNumber(self):
#         ability = str(random.randint(1, 10))
#         self.updateDisplay(ability)
#
#     def updateDisplay(self, myString):
#         self.displayVariable.set(myString)
#
#     def manipulateFont(self, fontSize=None, *args):
#         newFont = (font.get(), fontSize.get())
#         return newFont
#
#     def clear(self):
#         txt.delete(0, 'end')
#         res = ""
#         self.message.configure(text=res)
#
#     def clear2(self, txt2=None):
#         txt2.delete(0, 'end')
#         res = ""
#         self.message.configure(text=res)
#
#     def is_number(self, s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             pass
#
#         try:
#             import unicodedata
#             unicodedata.numeric(s)
#             return True
#         except (TypeError, ValueError):
#             pass
#
#         return False
#
#     def collectUserImageForRegistration(self):
#         clientIDVal = (self.clientIDTxt.get())
#         empIDVal = self.empIDTxt.get()
#         name = (self.empNameTxt.get())
#         ap = argparse.ArgumentParser()
#
#         ap.add_argument("--faces", default=50,
#                         help="Number of faces that camera will get")
#         ap.add_argument("--output", default="../datasets/train/" + name,
#                         help="Path to faces output")
#
#         args = vars(ap.parse_args())
#
#         trnngDataCollctrObj = TrainingDataCollector(args)
#         trnngDataCollctrObj.collectImagesFromCamera()
#
#         notifctn = "We have collected " + str(args["faces"]) + " images for training."
#         self.message.configure(text=notifctn)
#
#     def getFaceEmbedding(self):
#
#         ap = argparse.ArgumentParser()
#
#         ap.add_argument("--dataset", default="../datasets/train",
#                         help="Path to training dataset")
#         ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings2.pickle")
#         # Argument of insightface
#         ap.add_argument('--image-size', default='112,112', help='')
#         ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
#         ap.add_argument('--ga-model', default='', help='path to load model.')
#         ap.add_argument('--gpu', default=0, type=int, help='gpu id')
#         ap.add_argument('--det', default=0, type=int,
#                         help='mtcnn option, 1 means using R+O, 0 means detect from begining')
#         ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
#         ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
#         args = ap.parse_args()
#
#         genFaceEmbdng = GenerateFaceEmbedding(args)
#         genFaceEmbdng.genFaceEmbedding()
#
#     def trainModel(self):
#         # ============================================= Training Params ====================================================== #
#
#         ap = argparse.ArgumentParser()
#
#         # ap = argparse.ArgumentParser()
#         ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings2.pickle",
#                         help="path to serialized db of facial embeddings")
#         ap.add_argument("--model", default="faceEmbeddingModels/my_model2.h5",
#                         help="path to output trained model")
#         ap.add_argument("--le", default="faceEmbeddingModels/le2.pickle",
#                         help="path to output label encoder")
#
#         args = vars(ap.parse_args())
#
#         self.getFaceEmbedding()
#         faceRecogModel = TrainFaceRecogModel(args)
#         faceRecogModel.trainKerasModelForFaceRecognition()
#
#         notifctn = "Model training is successful.No you can go for prediction."
#         self.message.configure(text=notifctn)
#
#     def makePrediction(self):
#         faceDetector = FacePredictor()
#         faceDetector.detectFace()
#
#     def close_window(self):
#         self.window.destroy()
#
#     def callback(self, url):
#         webbrowser.open_new(url)
#
#
#     # # CODE ADDED FOR FLASK
#     # @app.route('/takeImages', methods=['POST'])
#     # def collectUserImageForRegistration(self):
#     #     print("Camera is opened")
#
#
# logFileName = "ProceduralLog.txt"
# regStrtnModule = RegistrationModule(logFileName)
#
# # regStrtnModule = RegistrationModule
# # regStrtnModule.TrainImages()















# CODE ADDED FOR FLASK

# @app.route('/takeImages', methods=['POST'])
# def take_images():
#     print("Camera is opened")
#     # You can call the necessary functions here
#     return {"message": "Camera is opened"}
#
# if __name__ == '__main__':
#     # logFileName = "ProceduralLog.txt"
#     # regStrtnModule = RegistrationModule(logFileName)
#     # # app.run(debug=True)
#     app.run(debug=True, port=8080)




# @app.route('/takeImages', methods=['POST'])
# def take_images():
#     print("Camera is opened")
#     # You can call the necessary functions here
#     return {"message": "Camera is opened"}
#
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)

# @app.route('/takeImages', methods=['POST'])
# def take_images():
#     # DEBUGGING
#     print("Camera is opened")
#
#     # Create the directory if it doesn't exist
#     directory = "../datasets/nonProcessed"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     # Save the incoming image to the directory
#     # DEBUGGING PURPOSE
#     print(request.files)  # Add this line
#
#     image = request.files['photo']
#     filename = secure_filename(image.filename)
#     image.save(os.path.join(directory, filename))
#
#     return {"message": "Image received and saved"}
#
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)




#
# @app.route('/takeImages', methods=['POST'])
# def take_images():
#     # DEBUGGING
#     print("Camera is opened")
#
#     # Create the directory if it doesn't exist
#     directory = "../datasets/nonProcessed"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     # Get the base64 string from the request
#     image_data = request.form['photo']
#     # DEBUGGING
#     print(f"IMAGE DATA: {image_data}")
#
#     # Remove the 'data:image/png;base64,' part from the string
#     image_data = image_data.split(',')[1]
#
#     # Decode the base64 string into bytes
#     image_bytes = base64.b64decode(image_data)
#
#     # Open a new image file and save the decoded bytes into this file
#     image = Image.open(io.BytesIO(image_bytes))
#     image.save(os.path.join(directory, 'image.jpg'))
#
#     return {"message": "Image received and saved"}




#
# from werkzeug.utils import secure_filename
# @app.route('/takeImages', methods=['POST'])
# def take_images():
#     # Check if the post request has the file part
#
#     if 'photo' not in request.files:
#         return {"message": "No file part in the request."}, 400
#
#     file = request.files['photo']
#
#     # If the user does not select a file, the browser might
#     # submit an empty file without a filename.
#     if file.filename == '':
#         return {"message": "No selected file."}, 400
#
#     # Create the directory if it doesn't exist
#     directory = "../datasets/nonProcessed"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(directory, filename))
#
#     # DEBUGGING
#     print("Image received and saved")
#
#     return {"message": "Image received and saved"}
#
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)







import os
from flask import request
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify

@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Client request received")
    student_name = request.form.get('studentName').strip()
    print(f"Student's name: {student_name}")

    if 'video' not in request.files:
        return {"message": "No video file part in the request."}, 400

    video_file = request.files['video']
    if video_file.filename == '':
        return {"message": "No selected file in the request."}, 400

    # Create directories if they don't exist
    input_directory = f"../datasets/nonProcessed/{student_name}"
    output_directory = f"../datasets/processed/{student_name}"

    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the video file
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(input_directory, filename)
    video_file.save(video_path)

    print("Video saved successfully")

    # Process the video (if needed)
    # You can add your video processing logic here

    return {"message": "Video uploaded successfully"}, 200









from werkzeug.utils import secure_filename

from PIL import Image, ExifTags

from keras import backend as K
import tensorflow as tf

import os
import json


# Create a global variable for the graph
global graph
graph = tf.get_default_graph()

# Create a global session
global session
session = K.get_session()


from PIL import Image, ExifTags
import os

from PIL import Image
import os

def correct_image_orientation(filename):
    try:
        image = Image.open(filename)
        # Rotate the image 90 degrees counter-clockwise
        image = image.rotate(-90, expand=True)

        # Resize the image while maintaining aspect ratio
        max_size = (1000, 1000)
        image.thumbnail(max_size, Image.ANTIALIAS)


        image.save(filename)
        image.close()

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

def correct_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                correct_image_orientation(os.path.join(root, file))


# def correct_image_orientation(filename):
#     try:
#         image=Image.open(filename)
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation]=='Orientation':
#                 break
#         exif=dict(image._getexif().items())
#
#         if exif[orientation] == 3:
#             image=image.rotate(180, expand=True)
#         elif exif[orientation] == 6:
#             image=image.rotate(270, expand=True)
#         elif exif[orientation] == 8:
#             image=image.rotate(90, expand=True)
#         image.save(filename)
#         image.close()
#
#     except (AttributeError, KeyError, IndexError):
#         # cases: image don't have getexif
#         pass
#
# def correct_images_in_directory(directory):
#     print("CORRECTING ORIENTATION")
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 correct_image_orientation(os.path.join(root, file))


from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

from flask import Flask, request, jsonify


@app.route('/takeImages', methods=['POST'])
def take_images():
    print("CLIENT request received")
    student_name = request.form.get('studentName')
    print(f"Student's first name: {student_name}")

    if 'photos' not in request.files:
        return {"message": "No file part in the request."}, 400

    files = request.files.getlist('photos')
    print("FILES", files)
    # student_name = request.form.get('studentName')
    student_name = request.form.get('studentName').strip()
    print(f"Student's first name: {student_name}")

    input_directory = f"../datasets/nonProcessed/{student_name}"
    output_directory = f"../datasets/processed/{student_name}"

    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # for file in files:
    #     if file:
    #         filename = secure_filename(file.filename)
    #         print("FILENAME", filename)
    #         file.save(os.path.join(input_directory, filename))
    #
    #         with graph.as_default():
    #             collector = TrainingDataCollector1(input_directory, output_directory, filename)
    #             collector.collectImagesFromCamera()


    # Save all images first
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            print("FILENAME", filename)
            file.save(os.path.join(input_directory, filename))

    # After all images have been uploaded and saved, correct the orientation
    # correct_images_in_directory(input_directory)
    #
    # # Process all images after they have been saved
    # with graph.as_default():
    #     print("INSIDE IMAGE PREPROCESSING") #debug
    #
    #     for file in files:
    #         if file:
    #             print("INSIDE FILE", file.filename) #debug
    #
    #             filename = secure_filename(file.filename)
    #             collector = TrainingDataCollector1(input_directory, output_directory, filename)
    #             collector.collectImagesFromCamera()
    #
    #
    # print("Images received, uploaded and saved and processed")

    # Start a background task for correcting image orientation
    # executor.submit(correct_images_in_directory, input_directory)
    future = executor.submit(correct_images_in_directory, input_directory)

    # Wait for the first task to complete before starting the second
    future.result()


    # Start a background task for image processing
    executor.submit(process_images, files, input_directory, output_directory)


    print("IMAGES UPLOADED SUCCESS")
    return {"message": "success"}


def process_images(files, input_directory, output_directory):
    with graph.as_default():
        print("INSIDE IMAGE PREPROCESSING") #debug

        for file in files:
            if file:
                print("INSIDE FILE", file.filename) #debug

                filename = secure_filename(file.filename)
                collector = TrainingDataCollector1(input_directory, output_directory, filename)
                collector.collectImagesFromCamera()

    print("Images received, saved and processed")


# def take_images():
#     if 'photos' not in request.files:
#         return {"message": "No file part in the request."}, 400
#
#     files = request.files.getlist('photos')
#     student_name = request.form.get('studentName')
#     print(f"Student's first name: {student_name}")
#
#     input_directory = f"../datasets/nonProcessed/{student_name}"
#     output_directory = f"../datasets/processed/{student_name}"
#
#     if not os.path.exists(input_directory):
#         os.makedirs(input_directory)
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     for file in files:
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(input_directory, filename))
#
#             with graph.as_default():
#                 collector = TrainingDataCollector1(input_directory, output_directory, filename)
#                 collector.collectImagesFromCamera()
#
#     print("Images received, saved and processed")
#     return {"message": "success"}


# def take_images():
#     # Check if the post request has the file part
#     if 'photo' not in request.files:
#         return {"message": "No file part in the request."}, 400
#
#     file = request.files['photo']
#     student_name = request.form.get('studentName')  # Get the student's first name from the form data
#
#     print(f"Student's first name: {student_name}")  # Print the student's first name for debugging
#
#     # If the user does not select a file, the browser might
#     # submit an empty file without a filename.
#     if file.filename == '':
#         return {"message": "No selected file."}, 400
#
#     # Create the directory if it doesn't exist
#     input_directory = f"../datasets/nonProcessed/{student_name}"
#     output_directory = f"../datasets/processed/{student_name}"
#     # directory = f"../datasets/nonProcessed/{student_name}"
#
#     if not os.path.exists(input_directory):
#         os.makedirs(input_directory)
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     # if not os.path.exists(directory):
#     #     os.makedirs(directory)
#
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(input_directory, filename))
#
#         # Use the global graph in this thread
#         with graph.as_default():
#             # Create an instance of TrainingDataCollector
#             collector = TrainingDataCollector1(input_directory, output_directory, filename)
#             # Call the collectImagesFromCamera method
#             collector.collectImagesFromCamera()
#
#     print("Image received, saved and processed")
#
#     return {"message": "success"}



# class for holding arguements
class Args:
    def __init__(self, dataset, embeddings, image_size, embedding_model, ga_model, gpu, det, flip, threshold, le, training_model):
        self.dataset = dataset
        self.embeddings = embeddings
        self.image_size = image_size
        # self.model = model
        self.embedding_model = embedding_model
        self.ga_model = ga_model
        self.gpu = gpu
        self.det = det
        self.flip = flip
        self.threshold = threshold
        self.le = le
        self.training_model = training_model

# Now you can access the arguments like this: args.dataset



# New route for initiating model training
@app.route('/startModelTraining', methods=['POST'])
def start_model_training():

    # CODE ADDED FOR TEMPORARY PURPOSE
    # START

    print("PREPROCESSING THE IMAGES START")

    # print("PREPROCESSING THE VIDEOS START")

    # Get the list of student names (which are the names of the folders in /datasets/nonProcessed)
    student_names = os.listdir("../datasets/nonProcessed")

    # For each student, preprocess their images
    # For each student, preprocess their videos
    for student_name in student_names:
        # Define the input and output directories for this student
        input_dir = f"../datasets/nonProcessed/{student_name}"
        output_dir = f"../datasets/processed/{student_name}"

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a TrainingDataCollector1 instance for this student
        # collector = TrainingDataCollector2(
        #     input_dir=input_dir,
        #     output_dir=output_dir,
        #     graph=graph,
        #     session=session
        # )

        collector = TrainingDataCollector3(
            input_dir=input_dir,
            output_dir=output_dir,
            graph=graph,
            session=session
        )

        # # Get the list of image files for this student
        # image_files = os.listdir(collector.input_dir)

        # Get the list of video files for this student
        video_files = os.listdir(collector.input_dir)

        # # Preprocess each image file
        # for image_file in image_files:
        #     # collector.image_file = image_file
        #     collector.collectImagesFromCamera(image_file)

        # Preprocess each video file
        for video_file in video_files:
            collector.collectImagesFromVideo(video_file)





    # END
    print("PREPROCESSING THE IMAGES ENDED")


    print("INSIDE THE TRAINING ROUTE API")

    # Create an Args object with your arguments
    args = Args(
        dataset="../datasets/processed",
        embeddings="faceEmbeddingModels/embeddings.pickle",
        image_size='112,112',
        embedding_model='../insightface/models/model-y1-test2/model,0',
        ga_model='',
        gpu=0,
        det=0,
        flip=0,
        threshold=1.24,
        le="faceEmbeddingModels/le.pickle",
        training_model="faceEmbeddingModels/my_model.h5"
    )

    # # Generate face embeddings
    # print("FACE EMBEDDING START")
    # # genFaceEmbdng = GenerateFaceEmbedding(args)
    # # genFaceEmbdng.genFaceEmbedding()
    # print("FACE EMBEDDING END")

    # Generate face embeddings
    print("FACE EMBEDDING START")
    with graph.as_default():
        K.set_session(session)
        genFaceEmbdng = GenerateFaceEmbedding(args)
        genFaceEmbdng.genFaceEmbedding()
    print("FACE EMBEDDING END")

    # # Train the model
    # print("MODEL TRAINING START")
    # faceRecogModel = TrainFaceRecogModel(args)
    # faceRecogModel.trainKerasModelForFaceRecognition()
    # print("MODEL TRAINING END")

    # Train the model
    print("MODEL TRAINING START")
    with graph.as_default():
        K.set_session(session)
        faceRecogModel = TrainFaceRecogModel(args, graph, session)
        faceRecogModel.trainKerasModelForFaceRecognition()
    print("MODEL TRAINING END")

    # # After model creation
    # global graph
    # graph = tf.get_default_graph()


    print("request received from admin web app")
    return jsonify({"message": "Model training initiated successfully!"})


# API END Point for Prediction
from threading import Thread
def start_prediction():
    print("BEGINNING THE  PREDICTION THREAD")
    global graph
    with graph.as_default():
        # Create a FacePredictor object and start prediction
        facePredictor = FacePredictor()
        facePredictor.detectFace()
    print("ENDING THE  PREDICTION THREAD")



@app.route('/startPrediction', methods=['POST'])
def handle_start_prediction_request():
    print("Prediction Start API End Point")
    # Start the prediction in a new thread

    print("OUTSIDE THE PREDICTION THREAD")
    Thread(target=start_prediction).start()
    return jsonify({"message": "Prediction started!"})


import psycopg2

@app.route('/updateDatabase', methods=['POST'])
def handle_update_database_request():
    # Extract the name from the request
    name = request.json['name']

    print(name + "will be recorded in database")

    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname="attendancesystem",
        user="postgres",
        password="password",
        host="localhost",
        port="5432"
    )

    # Create a new cursor object
    cur = conn.cursor()

    # Write your SQL query to insert a new attendance record
    query = """
    INSERT INTO Attendance3 (Student_ID, Attendance_Timestamp, Attendance_Status)
    VALUES (
        (SELECT Student_ID FROM Student WHERE Student_Name = %s),
        CURRENT_TIMESTAMP,
        TRUE
    );
    """

    # Execute the SQL query
    cur.execute(query, (name,))

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    return jsonify({"message": "Database updated!"})






if __name__ == '__main__':
    # app.run(debug=True, port=8080)

    # app.run(host='192.168.101.2', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)





