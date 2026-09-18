import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
from logger import MultiLogger
########################## FUNCTIONS OUTSIDES OF THE CLASS BUT VERY USEFUL ####################


def set_plot_size(BIGGER_SIZE):
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

def create_folder(folder_path):
    try:
    # creating a folder named data
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # if not created then raise error
    except OSError:
            print ('Error: Creating directory '+folder_path)

def get_datetime_from_file_name(video_name):
    try:
        name = video_name[:-4] if str(video_name).lower().endswith('.mp4') else str(video_name)

        # Pattern A: ..._YYYYMMDD_HHMMSS (optional _vNN)
        m = re.search(r'(\d{8})_(\d{6})(?:_v(\d+))?(?:_|$)', name)
        if m:
            ymd = m.group(1)
            hms = m.group(2)
            vnum = int(m.group(3)) if m.group(3) else 1
            base = datetime(
                int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]),
                int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
            )
            return base + timedelta(minutes=20 * (vnum - 1))

        # Pattern B: ..._YYMMDD_<label>_HHMMSS_vNN
        m = re.search(r'(\d{6})_[^_]+_(\d{6})_v(\d+)(?:_|$)', name)
        if m:
            ymd = m.group(1)
            hms = m.group(2)
            vnum = int(m.group(3))
            base = datetime(
                2000 + int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6]),
                int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
            )
            return base + timedelta(minutes=20 * (vnum - 1))

        # Pattern C: legacy camera format with explicit token
        s = name.find('_raspberrypi_')
        if s != -1:
            YY = int(name[s-6:s-4])
            MM = int(name[s-4:s-2])
            DD = int(name[s-2:s])
            HH = int(name[s+13:s+15])
            MI = int(name[s+15:s+17])
            SS = int(name[s+17:s+19])
            VV = int(name[s+21:s+23]) * 1204
            return datetime(2000 + YY, MM, DD, HH, MI, SS) + timedelta(seconds=VV)

        print("Incorrect name file, cannot find date")
        return None
    except Exception:
        print("Incorrect name file, cannot find date")
        return None

def progress_bar(current, total, bar_length=20):
    """Display progress bar. Only prints once per 10% step (plus 0% and 100%) to reduce console spam.

    A step is detected by the 10% bucket changing, not by ``percent % 10 == 0``: callers that report
    at a coarse stride never land exactly on a multiple of 10 (segmentation reports every 1000 of
    30060 frames -> 3%, 6%, 9%, 13%, ...), so they used to log only "0%" for their whole run, and the
    app showed a working segment as frozen. Display-only -- nothing reads this except the progress
    panels; loops that step 1% at a time print exactly the same lines as before.
    """
    fraction = current / total
    percent = int(fraction * 100)

    if not hasattr(progress_bar, 'last_percent'):
        progress_bar.last_percent = -1
    last = progress_bar.last_percent
    new_step = current == 0 or last < 0 or percent < last or percent // 10 != last // 10
    if new_step or current == total:
        if percent != last or current == total:
            progress_bar.last_percent = percent
            arrow = int(fraction * bar_length - 1) * '-' + '>'
            padding = int(bar_length - len(arrow)) * ' '
            message = f'Progress: [{arrow}{padding}] {percent}%'
            print(message)

#def progress_bar(logger, current, total, bar_length=20):
#    logger.progress(current, total, bar_length)


def get_ids_from_dict(resting_object_traj):
    list_of_columns_names = list(resting_object_traj.keys())
    list_of_ID_still = []
    for col_str in list_of_columns_names:
        if col_str.startswith("ID"):
            str_ID = col_str.partition("_")[2]
            list_of_ID_still.append(int(str_ID))
    return list_of_ID_still


def draw_parallelogram(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a copy of the image to draw the parallelogram on
    image_copy = image.copy()
    
    # Initialize a list to store the clicked points
    points = []
    points_list = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point to the list
            points.append((x, y))
            points_list.append([x,y])
            
            # Draw a circle at the clicked point
            cv2.circle(image_copy, (x, y), 3, (0, 255, 0), -1)
            
            # Display the image with the clicked points
            cv2.imshow("Image", image_copy)
            
            # Check if four points have been clicked
            if len(points) == 4:
                # Draw the parallelogram on the image
                cv2.line(image_copy, points[0], points[1], (0, 255, 0), 2)
                cv2.line(image_copy, points[1], points[2], (0, 255, 0), 2)
                cv2.line(image_copy, points[2], points[3], (0, 255, 0), 2)
                cv2.line(image_copy, points[3], points[0], (0, 255, 0), 2)
                
                # Display the final image with the parallelogram
                cv2.imshow("Image", image_copy)
                
                # Print the coordinates of the four points
                for i, point in enumerate(points):
                    print(f"Point {i+1}: {point}")
    
    # Create a window and set the mouse callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    
    # Display the image
    cv2.imshow("Image", image)
    
    # Wait for the user to close the window
    cv2.waitKey(0)
    
    # Close all windows
    cv2.destroyAllWindows()

    return points_list

def zero_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges