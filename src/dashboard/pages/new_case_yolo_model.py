import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolo11n.pt") 
names = model.model.names

cap = cv2.VideoCapture("/home/npnnpn_1984/hp1/hp/src/dashboard/components/assets/vid1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(300, 1000), (25000, 1000)]

speed_obj = solutions.SpeedEstimator(
    reg_pts = line_pts,
    names = names,
    view_img = True,
)

while cap.isOpened:
    success, im0 = cap.read()
    if not success:
        print("bla bla bla")
        break
    tracks = model.track(im0, persist =True)
    im0 = speed_obj.compute_speed(im0, tracks)
    #im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
