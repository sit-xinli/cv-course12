import cv2
import numpy as np
import sys

def main():
    # Check for video file argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
    else:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a grid of points to track (one point per 8x8 patch)
    h, w = old_gray.shape
    step = 8
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    p0 = np.vstack((x, y)).T.reshape(-1, 1, 2).astype(np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points and draw
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw the flow vector
                frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
                # Draw the tracked point
                frame = cv2.circle(frame, (int(a), int(b)), 2, (0, 0, 255), -1)
            
            cv2.imshow('Optical Flow', frame)

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # If all points are lost, re-initialize them
            if p0.shape[0] == 0:
                h, w = old_gray.shape
                step = 8
                y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
                p0 = np.vstack((x, y)).T.reshape(-1, 1, 2).astype(np.float32)


        # Press 'ESC' to exit
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
