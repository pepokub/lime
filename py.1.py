
import cv2
import numpy as np
import requests
import time
import os

# HSV color range for detecting green and yellow limes
low_g = np.array([30, 50, 50])
up_g = np.array([90, 255, 255])

low_y = np.array([20, 100, 100])
up_y = np.array([30, 255, 255])

pix_cm = 19
min_area = 1000
smooth = 0.1

# LINE Notify configuration
token = 'Wb3dCy9EMqyuQkoQrqjS0gDqnTYdQZVGZlbv3lBX68d'
notify_url = 'https://notify-api.line.me/api/notify'

def send_msg(msg, img_path):
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'message': msg}
    files = {'imageFile': open(img_path, 'rb')}
    try:
        res = requests.post(notify_url, headers=headers, data=payload, files=files)
        return res.status_code, res.text
    except requests.exceptions.RequestException as e:
        print(f"Error sending LINE Notify message: {e}")
        return None, str(e)

def save_img(frame, cnt, dia, area, circ, lime_msg):
    cv2.putText(frame, f"Dia: {dia:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Area: {area:.2f} sq.cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Circ: {circ:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, lime_msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
   
    img_name = f'captured_images/lime_{cnt}.jpg'
    cv2.imwrite(img_name, frame)
    return img_name

# Prompt user for desired lime juice volume in liters
desired_liters = float(input("Enter the desired lime juice volume (in liters): "))

# Initialize the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

lime_count = 0
lime_juice_values = []  # Store lime juice amounts for averaging

while lime_count < 10:  # Process up to 10 limes
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, low_g, up_g),
        cv2.inRange(hsv, low_y, up_y)
    )

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            obj_size = round(w / pix_cm, 2)
            last_size = (1 - smooth) * obj_size + smooth * obj_size

            lime_juice = 0
            if 4 <= last_size <= 4.49:
                lime_juice = round(last_size * 2.85, 2)
            elif last_size >= 4.5:
                lime_juice = round(last_size * 3.91, 2)
            elif 3 <= last_size <= 3.9:
                lime_juice = round(last_size * 2.46, 2)

            lime_juice_values.append(lime_juice)  # Store the juice amount

            lime_msg = f"Lime juice: {lime_juice} ml" if lime_juice > 0 else ""

            center = (int(cx), int(cy))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

            dia = last_size
            area = np.pi * (last_size / 2) ** 2
            circ = 2 * np.pi * last_size

            lime_count += 1

            img_name = save_img(frame, lime_count, dia, area, circ, lime_msg)
            time.sleep(5)

            if lime_count < 10:
                msg = f"Lime {lime_count}: {lime_juice} ml"
            else:
                avg_juice = np.mean(lime_juice_values)  # Calculate average
                limes_needed = np.ceil(desired_liters * 1000 / avg_juice)  # Calculate limes needed
                msg = (
                    f"Average lime juice: {avg_juice:.2f} ml\n"
                    f"Limes needed: {limes_needed:.0f}"
                )

            send_msg(msg, img_name)

    cv2.imshow("Lime Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()