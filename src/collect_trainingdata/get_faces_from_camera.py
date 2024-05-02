import sys

# from src.insightface.src.common import face_preprocess
from src.insightface.src.common import face_preprocess
# from insightface.src.common import face_preprocess

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
# import face_preprocess
import numpy as np
import cv2
import os
from datetime import datetime

from keras import backend as K


class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        # Detector = mtcnn_detector
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        # initialize video stream
        cap = cv2.VideoCapture(0)

        # Setup some useful var
        faces = 0
        frames = 0
        max_faces = int(self.args["faces"])
        max_bbox = np.zeros(4)

        if not (os.path.exists(self.args["output"])):
            os.makedirs(self.args["output"])

        while faces < max_faces:    # HERE WE ARE COLLECTING THE FRAMES FOR IMAGES
            ret, frame = cap.read()
            frames += 1

            dtString = str(datetime.now().microsecond)
            # Get all faces on current frame
            bboxes = self.detector.detect_faces(frame)

            if len(bboxes) != 0:
                # Get only the biggest face
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3 frames
                if frames % 3 == 0:
                    # convert to face_preprocess.preprocess input
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg)
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




# KEEPING THE OLD CLASS AND CREATING A SIMILAR NEW ONE
#
# class TrainingDataCollector1:
#     def __init__(self, input_dir, output_dir):
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         self.detector = MTCNN()
#
#     def collectImagesFromCamera(self):
#         # Get the list of image files in the input directory
#         image_files = os.listdir(self.input_dir)
#
#         for image_file in image_files:
#             # Read the image file
#             frame = cv2.imread(os.path.join(self.input_dir, image_file))
#
#             # Get all faces on current frame
#             bboxes = self.detector.detect_faces(frame)
#
#             if len(bboxes) != 0:
#                 # Get only the biggest face
#                 max_area = 0
#                 for bboxe in bboxes:
#                     bbox = bboxe["box"]
#                     bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
#                     keypoints = bboxe["keypoints"]
#                     area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#                     if area > max_area:
#                         max_bbox = bbox
#                         landmarks = keypoints
#                         max_area = area
#
#                 max_bbox = max_bbox[0:4]
#
#                 # convert to face_preprocess.preprocess input
#                 landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
#                                       landmarks["mouth_left"][0], landmarks["mouth_right"][0],
#                                       landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
#                                       landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
#                 landmarks = landmarks.reshape((2, 5)).T
#                 nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
#
#                 # Save the preprocessed image to the output directory
#                 cv2.imwrite(os.path.join(self.output_dir, image_file), nimg)




class TrainingDataCollector1:
    def __init__(self, input_dir, output_dir, image_file):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_file = image_file
        self.detector = MTCNN()

    def collectImagesFromCamera(self):

        print("BEGINNING OF  collectImageFromCamera function ")

        # Read the image file
        frame = cv2.imread(os.path.join(self.input_dir, self.image_file))

        # Get all faces on current frame
        bboxes = self.detector.detect_faces(frame)

        if len(bboxes) != 0:
            # Get only the biggest face
            max_area = 0
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                keypoints = bboxe["keypoints"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_bbox = bbox
                    landmarks = keypoints
                    max_area = area

            max_bbox = max_bbox[0:4]

            # convert to face_preprocess.preprocess input
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                  landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                  landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                  landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2, 5)).T
            print("ABOUT TO CALL FACE PREPROCESS FUNCTION")
            nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

            # Save the preprocessed image to the output directory
            print("END OF collectImageFromCamera function ")
            cv2.imwrite(os.path.join(self.output_dir, self.image_file), nimg)




class TrainingDataCollector2:
    def __init__(self, input_dir, output_dir, graph, session):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.graph = graph
        self.session = session

    def collectImagesFromCamera(self, image_file):
        self.image_file = image_file
        with self.graph.as_default():
            K.set_session(self.session)
            self.detector = MTCNN()
            print("BEGINNING OF collectImageFromCamera function")

            # Read the image file
            frame = cv2.imread(os.path.join(self.input_dir, self.image_file))

            # Get all faces on current frame
            bboxes = self.detector.detect_faces(frame)

            if len(bboxes) != 0:
                # Get only the biggest face
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[0:4]

                # convert to face_preprocess.preprocess input
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                      landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                      landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                      landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2, 5)).T
                print("ABOUT TO CALL FACE PREPROCESS FUNCTION")
                nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                # Save the preprocessed image to the output directory
                print("END OF collectImageFromCamera function")
                cv2.imwrite(os.path.join(self.output_dir, self.image_file), nimg)










class TrainingDataCollector3:

    def __init__(self, input_dir, output_dir, graph, session):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.graph = graph
        self.session = session

    def collectImagesFromVideo(self, video_file):
        with self.graph.as_default():
            K.set_session(self.session)
            self.detector = MTCNN()

            # Open the video file
            cap = cv2.VideoCapture(os.path.join(self.input_dir, video_file))

            # Initialize frame counter
            frame_count = 0

            while True:
                # Read a frame from the video file
                ret, frame = cap.read()

                # If the frame was not read successfully, we're done
                if not ret:
                    break

                # Detect faces in the frame
                bboxes = self.detector.detect_faces(frame)

                # If any faces are detected, save the largest one
                if bboxes:
                    # Get only the biggest face
                    max_area = 0
                    for bboxe in bboxes:
                        bbox = bboxe["box"]
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        keypoints = bboxe["keypoints"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_bbox = bbox
                            landmarks = keypoints
                            max_area = area

                    max_bbox = max_bbox[0:4]

                    # convert to face_preprocess.preprocess input
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    # Save the preprocessed face image
                    cv2.imwrite(os.path.join(self.output_dir, f"{frame_count}.jpg"), nimg)

                    # Check the total number of images in the output directory
                    total_images = len(os.listdir(self.output_dir))
                    if total_images >= 200:
                        break

                # Increment the frame counter
                frame_count += 1

                # Release the frame from memory after processing
                del frame

            # Release the video file
            cap.release()





import cv2
import os
from mtcnn.mtcnn import MTCNN
# from src.insightface.src.common import face_preprocess
from collections import Counter


class TrainingDataCollector4:

    def __init__(self, input_dir, output_dir, graph, session):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.graph = graph
        self.session = session
        self.orientations = ["straight", "up", "right", "left", "down", "up_right", "down_right", "up_left", "down_left"]
        self.max_images_per_orientation = 30

    def collectImagesFromVideo(self, video_file):
        with self.graph.as_default():
            K.set_session(self.session)
            self.detector = MTCNN()

            # Open the video file
            cap = cv2.VideoCapture(os.path.join(self.input_dir, video_file))

            # Initialize counters for each orientation
            counts = {orientation: 0 for orientation in self.orientations}

            while True:
                # Read a frame from the video file
                ret, frame = cap.read()

                # If the frame was not read successfully, we're done
                if not ret:
                    break

                # Detect faces in the frame
                bboxes = self.detector.detect_faces(frame)

                # If any faces are detected, save the largest one
                if bboxes:
                    # Get only the biggest face
                    max_area = 0
                    for bboxe in bboxes:
                        bbox = bboxe["box"]
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        keypoints = bboxe["keypoints"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_bbox = bbox
                            landmarks = keypoints
                            max_area = area

                    max_bbox = max_bbox[0:4]

                    # convert to face_preprocess.preprocess input
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    # Determine the orientation of the face
                    orientation = self.determine_orientation(landmarks)

                    # If we haven't collected enough images for this orientation
                    if counts[orientation] < self.max_images_per_orientation:
                        # Save the preprocessed face image
                        cv2.imwrite(os.path.join(self.output_dir, f"{orientation}_{counts[orientation]}.jpg"), nimg)
                        counts[orientation] += 1

            # Release the video file
            cap.release()

    import numpy as np
    from collections import Counter

    def determine_orientation(self,landmarks_sequence):
        """
        Determine facial orientation from a sequence of facial landmarks.

        Parameters:
            landmarks_sequence (list): A list of dictionaries containing facial landmarks
                                       detected in each frame of the video clip.

        Returns:
            orientation (str): The estimated facial orientation for the entire video clip.
        """
        orientations = []

        for landmarks_frame in landmarks_sequence:
            # Extract landmark coordinates for eyes and nose
            left_eye = landmarks_frame['left_eye']
            right_eye = landmarks_frame['right_eye']
            nose_tip = landmarks_frame['nose_tip']

            # Calculate vectors representing eye positions and nose direction
            eye_vector = np.array(right_eye) - np.array(left_eye)
            nose_vector = np.array(nose_tip) - (np.array(left_eye) + np.array(right_eye)) / 2

            # Calculate angles between eye vector and nose vector
            angle = np.arccos(
                np.dot(eye_vector, nose_vector) / (np.linalg.norm(eye_vector) * np.linalg.norm(nose_vector)))
            angle_deg = np.degrees(angle)

            # Determine orientation based on the angle
            if angle_deg < 30:
                orientations.append("straight")
            elif 30 <= angle_deg < 60:
                orientations.append("up")
            elif 60 <= angle_deg < 120:
                orientations.append("left" if eye_vector[0] > 0 else "right")
            elif 120 <= angle_deg < 150:
                orientations.append("down")
            else:
                orientations.append("unknown")

        # Count occurrences of each orientation
        orientation_counter = Counter(orientations)

        # Get the most common orientation
        orientation = orientation_counter.most_common(1)[0][0]

        return orientation





