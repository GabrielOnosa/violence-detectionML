import cv2
import google.generativeai as genai
import os
import pandas as pd
from PIL import Image
import time
#steps : iterate through train directory
# for every video - iterate until frame 75 - extract it - give it to gemini
# get result.text from gemini, add it into a csv file along with the video name
import Yolo_Frame_selector
from ultralytics import YOLO

def Gemini_images_append(directory, csv_file = 'gemini_prompts_other_frame_val.csv'):
    #root_dir = r"D:\Downloads\database\RWF-2000\train"
    root_dir = directory
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        processed_videos = set(existing_df['Folder'].tolist())
    else:
        existing_df = pd.DataFrame()
        processed_videos = set()
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel("gemini-2.5-flash")
    i=1
    yolo_pose = YOLO('yolov8s-pose.pt')
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root_dir)
            label = os.path.basename(dirpath)
            if rel_path in processed_videos:
                    continue
            file_path = os.path.join(dirpath, filename)
            videoCap = cv2.VideoCapture(file_path)
            if not videoCap.isOpened():
                print("Error opening video stream or file")
                continue
            frames_to_capture = []
            frames_to_capture.append( Yolo_Frame_selector.yolo_frame_selector(file_path, yolo_pose)[0])
            for frame_number in frames_to_capture:
                videoCap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, image = videoCap.read()

                if success:
                    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    time.sleep(1)
                    response = model.generate_content([image_pil, "Describe what is happening in at most 30 words. Also, answer with just \"yes\" or \"no\" if there are signs of aggressive behaviour in this picture. "])
                    # results.append({"video_name": filename,
                    #                 "frame_number": frame_number,
                    #                 "response": response.text})
                    clean_text = " ".join(response.text.split())
                    df = pd.DataFrame({"video_name": [filename],
                                    "frame_number": [frame_number],
                                    "response": [clean_text],
                                       "Folder": [rel_path],
                                       "Label": [label]})
                    df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
                    print(f"Chosen frame: {frame_number}")
                    print(f"Prompt:{i}")
                    i+=1
                    print(response.text)
                else:
                    print(f"failed to capture frame on position {frame_number}")
            videoCap.release()
            cv2.destroyAllWindows()  # Closes any windows/pipelines


Gemini_images_append(r"D:\Downloads\database\RWF-2000\val")






