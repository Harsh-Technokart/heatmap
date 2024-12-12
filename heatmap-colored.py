import numpy as np
import cv2
import copy
import os
from make_video import make_video
from progress.bar import FillingSquaresBar


def main():
    
    capture = cv2.VideoCapture('test_2.mp4')
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        # creating a folder named data 
        if not os.path.exists('frames'): 
            os.makedirs('frames') 

    except OSError: 
        print('Error: Creating directory of data') 
        
    bar = FillingSquaresBar('Processing Frames', max=length)

    first_iteration_indicator = 1
    for i in range(0, length):

        ret, frame = capture.read()
        
        if not ret:
            break

        # If first frame
        if first_iteration_indicator == 1:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            filter = background_subtractor.apply(frame)  # remove the background

            threshold = 2
            maxValue = 2
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)

            # Normalize to scale between 0 and 255
            normalized_accum = cv2.normalize(accum_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create a custom colormap: blue -> yellow -> orange -> red
            color_map = np.zeros((256, 1, 3), dtype=np.uint8)
            for j in range(256):
                if j < 85:  # Map low intensity to blue
                    color_map[j] = [255 - (j * 3), j * 3, 255]
                elif j < 170:  # Map medium intensity to yellow/orange
                    color_map[j] = [0, 255, 255 - ((j - 85) * 3)]
                else:  # Map high intensity to red
                    color_map[j] = [0, 255 - ((j - 170) * 3), 255]

            # Apply the colormap using cv2.applyColorMap instead of cv2.LUT
            custom_colormap = cv2.applyColorMap(normalized_accum, color_map)

            # Overlay custom colormap on the first frame
            result_overlay = cv2.addWeighted(first_frame, 0.7, custom_colormap, 0.7, 0)

            name = "./frames/frame%d.jpg" % i
            cv2.imwrite(name, result_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.next()

    bar.finish()
    
    make_video('./frames/', './output.avi')

    # save the final heatmap
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()

