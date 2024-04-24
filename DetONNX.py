import torch
import cv2
import time

#model = torch.hub.load('.', 'custom', 'runs/train/exp15-tick/weights/best.pt', source='local', device='0' )
model = torch.hub.load('.', 'custom', 'runs/train/exp15-tick/weights/best.onnx', source='local', device='cpu' )
#model = torch.hub.load('.', 'custom', 'runs/train/exp15-tick/weights/best.torchscript', source='local', device='0' )
# Set device to GPU if available for hardware acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#cameraRight = cv2.VideoCapture(1)  # open the right camera
#cameraLeft = cv2.VideoCapture(0)   # open the left camera

cameraRight = cv2.VideoCapture('Testvideo/UitmSA.mp4')  # open the right camera
#cameraLeft = cv2.VideoCapture('Testvideo/UitmSA.mp4')   # open the left camera

if not cameraRight.isOpened():
    print("Cannot open right camera")

'''
if not cameraLeft.isOpened():
    print("Cannot open left camera")
'''     

while True:
    start_time = time.time()
    ret1, frameR = cameraRight.read()
    #ret2, frameL = cameraLeft.read()

    frameR = cv2.resize(frameR, (640, 640))
    #frameL = cv2.resize(frameL, (640, 640))

    #frame = cv2.hconcat([frameL,frameR])
    frame = frameR
    if frame is None:
        break
    result = model(frame)
    df = result.pandas().xyxy[0]

    # Convert coordinates to integers
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)

    # Concatenate label and confidence columns
    df['class'] = df['name'].astype(str)
    df['conf'] = df['confidence'].round(decimals=2)

    

    # Draw rectangles and put text on frame
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label = row['class'] + ' ' + str(row['conf'])
        conf_val = row['conf']
        if conf_val <= 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        if conf_val >= 0.7:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if conf_val < 0.7 and conf_val > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        print(row['class'] + '  :   ' + str(row['conf']))
        

    elapsed_time = time.time() - start_time
    cv2.putText(frame, f"FPS: {1/elapsed_time:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.waitKey(10)
