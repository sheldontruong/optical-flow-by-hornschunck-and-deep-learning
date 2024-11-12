import cv2
import numpy as np
from glob import glob

def compute_horn_schunck_optical_flow(frame1, frame2, alpha=0.001, num_iterations=100):
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Compute gradients
    Ix = cv2.Sobel(frame1_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(frame1_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = frame2_gray - frame1_gray

    # Initialize flow vectors (u, v) to zero
    u = np.zeros_like(frame1_gray)
    v = np.zeros_like(frame1_gray)

    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6, 0, 1/6],
                       [1/12, 1/6, 1/12]], dtype=np.float32)

    # Iterative optimization process
    for _ in range(num_iterations):
        # Compute local averages of the flow vectors
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)

        # Update flow vectors using the Horn-Schunck formula
        num = Ix * u_avg + Iy * v_avg + It
        den = alpha ** 2 + Ix ** 2 + Iy ** 2

        u = u_avg - (Ix * num) / den
        v = v_avg - (Iy * num) / den

    return u, v

def avg_magnitude(u,v,scale):
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg

# frame1 = cv2.imread('dataset/MiniCooper/frame07.png')
# frame2 = cv2.imread('dataset/MiniCooper/frame08.png')
cap = cv2.VideoCapture('dataset/slow_traffic_small.mp4')
# Get properties of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object for the output video
# For .mp4 output, use 'mp4v' codec. For .avi, you can use 'XVID'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
ret, frame1 = cap.read()
if not ret:
    print("Error reading first frame.")
    cap.release()
    exit()
# frames = glob('dataset/DogDance/*')
# frame1 = cv2.imread(frames[0])
# for frame2 in frames[1:]:
while True:
    ret, frame2 = cap.read()

    # Check if frame reading was successful
    if not ret:
        print("End of video or error reading frame.")
        break
    # frame2 = cv2.imread(frame2)
    # Compute optical flow
    u, v = compute_horn_schunck_optical_flow(frame1, frame2, alpha=15, num_iterations= 300)


    scale = 3
    avg_mag = avg_magnitude(u,v,scale)
    # Overlay flow vectors on the actual image
    overlay = frame1.copy()
    step = 8  # Step size for sampling points to reduce clutter
    for y in range(0, u.shape[0], step):
        for x in range(0, u.shape[1], step):
            # Get flow vector at each point
            fx, fy = u[y, x], v[y, x]
            # Calculate endpoint for the arrow based on flow vector
            end_x = int(x + fx * 100 * scale)  # Scale flow for visibility
            end_y = int(y + fy * 100 * scale)

            mag = magnitude = ((fx* scale)**2 + (fy* scale)**2)**0.5
            if mag > avg_mag:
                # Draw arrow on the overlay image
                cv2.arrowedLine(overlay, (x, y), (end_x, end_y), (255, 255, 0), 1, tipLength=0.1)
    frame1 = frame2.copy()
    # Display the result
    cv2.imshow("Optical Flow Overlay", overlay)
    out.write(overlay)
    cv2.waitKey(1)
out.release()
cap.release()
cv2.destroyAllWindows()