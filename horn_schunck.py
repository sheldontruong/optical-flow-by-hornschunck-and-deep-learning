import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_horn_schunck_optical_flow(frame1, frame2, alpha=0.001, num_iterations=100):
    # Convert frames to grayscale
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        frame1_gray, frame2_gray = frame1/255., frame2/255.

    # Compute gradients
    Ex = cv2.Sobel(frame1_gray, cv2.CV_64F, 1, 0, ksize=3)
    Ey = cv2.Sobel(frame1_gray, cv2.CV_64F, 0, 1, ksize=3)
    Et = frame2_gray - frame1_gray

    # Initialize flow vectors (u, v) to zero
    u = np.zeros_like(frame1_gray)
    v = np.zeros_like(frame1_gray)

    # Laplacian kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6, 0, 1/6],
                       [1/12, 1/6, 1/12]], 
                       dtype=np.float32)

    track_energy = []
    # Iterative optimization process
    for iter in tqdm(range(num_iterations)):
        # Compute local averages of the flow vectors
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)

        # Update flow vectors using the Horn-Schunck formula
        num = Ex * u_avg + Ey * v_avg + Et
        den = alpha ** 2 + Ex ** 2 + Ey ** 2

        u = u_avg - (Ex * num) / den
        v = v_avg - (Ey * num) / den

        du_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
        du_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
        dv_x = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
        dv_y = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)
        energy = ( Ex * u + Ey * v + Et )**2 + ( du_x**2 + du_y**2 + dv_x**2 + dv_y**2 )
        track_energy.append((iter, energy.sum()))

    iterations, losses = zip(*track_energy)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy over Iterations')
    plt.grid(True)
    plt.show()
    return u, v

def avg_magnitude(u,v, sample_step_size = 10):
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], sample_step_size):
        for j in range(0, u.shape[1], sample_step_size):
            counter += 1
            magnitude = (v[i,j]**2 + u[i,j]**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg
# def overlay_flow_on_image()

frame1 = cv2.imread('dataset/MiniCooper/frame07.png')
frame2 = cv2.imread('dataset/MiniCooper/frame08.png')
# image_size = (500, 500)
# circle_position_frame1 = (250, 250)  # Initial position of the circle (center)
# circle_position_frame2 = (250, 251)
# circle_radius = 100       # Larger radius for more prominent circle
# circle_intensity = 50   # Brightness of the circle

# # Generate frame 1 (noise image with a bright circle in the center)
# frame1 = np.random.rand(*image_size) * 100  # Higher intensity noise background
# frame2 = frame1.copy()
# # Adding a bright circle to frame1
# for i in range(image_size[0]):
#     for j in range(image_size[1]):
#         if (i - circle_position_frame1[0]) ** 2 + (j - circle_position_frame1[1]) ** 2 < circle_radius ** 2:
#             frame1[i, j] += circle_intensity  # Bright circle
# # Shift the entire circle region 1 pixel to the right in frame2
# for i in range(image_size[0]):
#     for j in range(image_size[1] - 1):  # Ensure no out-of-bounds shift
#         if (i - circle_position_frame1[0]) ** 2 + (j - circle_position_frame1[1]) ** 2 < circle_radius ** 2:
#             frame2[i, j + 1] = frame1[i, j]  # Move pixel to the right
# # Display the generated images
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(frame1, cmap='gray')
# ax[0].set_title("Frame 1 (Bright Circle in Noise)")
# ax[0].axis('off')

# ax[1].imshow(frame2, cmap='gray')
# ax[1].set_title("Frame 2 (Circle Shifted Right)")
# ax[1].axis('off')

# plt.show()


# cap = cv2.VideoCapture('dataset/slow_traffic_small.mp4')
# # Get properties of the input video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# frames = glob('dataset/Dumptruck/*')
# frames.sort()
# frame1 = cv2.imread(frames[0])
# print(frame1.shape)
# frame_width = int(frame1.shape[1])
# frame_height = int(frame1.shape[0])
# fps = int(20)
# Define the codec and create VideoWriter object for the output video
# For .mp4 output, use 'mp4v' codec. For .avi, you can use 'XVID'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
# # Check if the video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
# ret, frame1 = cap.read()
# if not ret:
#     print("Error reading first frame.")
#     cap.release()
#     exit()
outputs = []
# for frame2 in frames[1:]:
while True:
    # ret, frame2 = cap.read()

    # # Check if frame reading was successful
    # if not ret:
    #     print("End of video or error reading frame.")
    #     break
    # frame2 = cv2.imread(frame2)
    # Compute optical flow
    u, v = compute_horn_schunck_optical_flow(frame1, frame2, alpha=15, num_iterations= 300)


    viz_flow_scale = 200
    # Overlay flow vectors on the actual image
    overlay = frame1.copy()
    whiteim = np.ones_like(frame1) * 255
    step = 10  # Step size for sampling points to reduce clutter
    avg_mag = avg_magnitude(u,v,step)
    for y in range(0, u.shape[0], step):
        for x in range(0, u.shape[1], step):
            # Get flow vector at each point
            fx, fy = u[y, x], v[y, x]
            # Calculate endpoint for the arrow based on flow vector
            end_x = int(x + fx * viz_flow_scale)  # Scale flow for visibility
            end_y = int(y + fy * viz_flow_scale)

            mag = ((fx)**2 + (fy)**2)**0.5
            # if mag > avg_mag:
            #     # Draw arrow on the overlay image
            cv2.arrowedLine(overlay, (x, y), (end_x, end_y), (255, 255, 0), 1, tipLength=0.1)
            # cv2.arrowedLine(whiteim, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.1)
    frame1 = frame2.copy()
    # Display the result
    cv2.imshow("Optical Flow Overlay", overlay)
    # cv2.imshow("Optical Flow Overlay", whiteim)
    
    # outputs.extend([overlay]*10)
    cv2.waitKey(0)
    break
# for i in outputs:
#     out.write(i)
# out.release()
# cap.release()
# cv2.destroyAllWindows()