import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('ultralytics/yolov8n.pt')

# Open the video file
video_path = "ultralytics/videos/Milo.mp4"
cap = cv2.VideoCapture(video_path)
count = 0
skip = 5
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    count += 1
    if success:

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        if count % skip == 0:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            print(count)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()