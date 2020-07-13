import cv2
from time import sleep

cap = cv2.VideoCapture(0)

sleep(5)

print('Frame Width: ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Frame Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Frame Rate:  ', cap.get(cv2.CAP_PROP_FPS))
print('Brightness:  ', cap.get(cv2.CAP_PROP_BRIGHTNESS))
print('Contrast:    ', cap.get(cv2.CAP_PROP_CONTRAST))
print('Saturation:  ', cap.get(cv2.CAP_PROP_SATURATION))
print('Hue:         ', cap.get(cv2.CAP_PROP_HUE))
print('Gain:        ', cap.get(cv2.CAP_PROP_GAIN))
print('Exposure:    ', cap.get(cv2.CAP_PROP_EXPOSURE))


while True:
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()