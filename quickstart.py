from __future__ import print_function
import pickle
import cv2
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import httplib2
import io
import http
from googleapiclient.http import MediaIoBaseDownload
import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# If modifying these scopes, delete the file token.pickle.
SCOPES = 'https://www.googleapis.com/auth/drive'


def PIL2array(img):
    """ Convert a PIL/Pillow image to a numpy array """
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def up_sample(landmark_list, sample_size=4):
    for face_landmark in landmark_list:
        if len(face_landmark) > 1:
            for key in face_landmark.keys():
                face_landmark[key] = [(w[0] * sample_size, w[1] * sample_size) for w in face_landmark[key]]
    return landmark_list


class FaceLandMarkDetection:

    def predict(self, frame):
        self.face_landmarks = face_recognition.face_landmarks(frame)

    #         if down_sampling:
    #             self.face_landmarks = up_sample(face_landmarks)
    #         else:
    #             self.face_landmarks = face_landmarks

    def plot(self, frame):
        pil_image = Image.fromarray(frame)
        print(self.face_landmarks)
        for face_landmarks in self.face_landmarks:
            if len(face_landmarks) > 1:
                d = ImageDraw.Draw(pil_image, 'RGB')

                # Make the eyebrows into a nightmare
                d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39))
                d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39))
                d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39), width=5)
                d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39), width=5)

                # Gloss the lips
                d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0))
                d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0))
                d.line(face_landmarks['top_lip'], fill=(150, 0, 0), width=8)
                d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0), width=8)

                # Apply some eyeliner
                d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0), width=6)
                d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0), width=6)
                d.polygon(face_landmarks['nose_tip'], fill=(150, 0, 0))

        return PIL2array(pil_image)



def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId='1ALXmFWrnaV7vhCDMPW2N4sPudTsqpjr-')
    fh = io.FileIO('this.jpg','wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    img = plt.imread("this.jpg")



    face_landmark_detection = FaceLandMarkDetection()
    face_landmark_detection.predict(img)
    l = face_landmark_detection.plot(img)
    im = Image.fromarray(l)
    im.save("output.png")

if __name__ == '__main__':
    main()