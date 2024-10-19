import cv2
import numpy as np
import time
from fastapi import WebSocket

# Initialize variables
start_time = None
previous_character = None


async def process_video(websocket: WebSocket):
    print("Accepting WebSocket connection...")
    await websocket.accept()
    time.sleep(2)
    print("WebSocket connection accepted.")

    previous_character = None
    cap = cv2.VideoCapture(0)  # Capture video from the default camera

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        # Resize the frame to make it smaller (250x200)
        frame = cv2.resize(frame, (250, 200))

        # Convert the frame to JPEG
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            break

        # Convert the JPEG image to bytes and send it to the client
        frame_bytes = jpeg.tobytes()
        await websocket.send_bytes(frame_bytes)

        # Process the frame for hand gesture recognition
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect hand landmark data for prediction
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Define bounding box for hand landmarks
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    # Make prediction for this hand
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    if predicted_character != previous_character:
                        start_time = time.time()
                        previous_character = predicted_character

                    # If the sign remains the same for at least 1 second, print it
                    elif start_time is not None and time.time() - start_time >= 1:
                        print("Predicted Hand Sign:", predicted_character)
                        start_time = None
                        # Send the predicted character as a plain string
                        await websocket.send_text(predicted_character)

        except Exception as e:
            # Handle any unexpected errors gracefully
            continue  # Continue processing the next frame

        # Send the processed frame bytes to the WebSocket client
        ret, jpeg = cv2.imencode(".jpg", frame)
        if ret:
            frame_bytes = jpeg.tobytes()
            await websocket.send_bytes(frame_bytes)


# WebSocket endpoint for video feed
@app.websocket("/video-feed")
async def video_feed(websocket: WebSocket):
    await process_video(websocket)
