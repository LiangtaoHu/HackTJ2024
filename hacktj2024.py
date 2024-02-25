import cv2
import numpy as np
from ultralytics import YOLO
import pygame
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import open3d as o3d

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cam = cv2.VideoCapture(0)
yolomodel = YOLO("yolov8x-seg.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
captionmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

ret, frame = cam.read()
frame = cv2.resize(frame, (1280, 720))
pygame.init()
#make pygame window the same size as the frame, not fullscreen
pygame.display.set_mode(frame.shape[1::-1], pygame.RESIZABLE)
while True:
    ret, orgframe = cam.read()
    orgframe = cv2.resize(orgframe, (1280, 720))
    input_batch = transform(orgframe).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=orgframe.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    # Display the depth map using OpenCV
    output = prediction.cpu().numpy()
    output = output / output.max()
    cv2.imshow("depth", output)
    #convert output to rgb image
    rgboutput = cv2.applyColorMap(cv2.convertScaleAbs(output, alpha=255), cv2.COLORMAP_JET)
    #detect objects in the frame
    results = yolomodel.predict(orgframe, imgsz=736, verbose=False)
    #draw the bounding boxes
    frame = results[0].plot()
    frame = np.rot90(frame, 1)
    frame = np.flip(frame, 0)
    #display the frame using pygame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = pygame.surfarray.make_surface(frame)
    pygame.display.get_surface().blit(frame, (0, 0))
    pygame.display.flip()
    #if the user clicks on a detection, print the class and confidence
    if pygame.mouse.get_pressed()[0]:
        x1, y1 = pygame.mouse.get_pos()
        for b, m in zip(results[0].boxes, results[0].masks):
            if cv2.pointPolygonTest(np.array(m.xy), (x1, y1), False) > 0:
                xyxy = b.xyxy.tolist()[0]
                # print(yolomodel.names[b.cls.tolist()[0]], b.conf.tolist()[0])
                croppedimage = orgframe[max(int(xyxy[1])-50, 0):min(int(xyxy[3])+50, orgframe.shape[0]), max(int(xyxy[0])-50, 0):min(int(xyxy[2]+50), orgframe.shape[1])]
                cv2.imshow("object", croppedimage)
                cv2.imshow("object depth", output[max(int(xyxy[1])-50, 0):min(int(xyxy[3])+50, orgframe.shape[0]), max(int(xyxy[0])-50, 0):min(int(xyxy[2]+50), orgframe.shape[1])])
                text = "there is a " + yolomodel.names[b.cls.tolist()[0]]
                inputs = processor(croppedimage, text, return_tensors="pt").to("cuda")
                out = captionmodel.generate(**inputs)
                print(processor.decode(out[0], skip_special_tokens=True))

                #plot 3d point cloud of the selected object using open3d using the depth map and the bounding box
                # Prepare mesh grid
                height, width = orgframe.shape[:2]
                x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

                # Reproject depth into 3D space
                fx, fy = 1000, 1000  # Focal length
                cx, cy = width // 2, height // 2  # Principal point (center of the image)
                z = output
                x = (x_grid - cx) * z / fx
                y = (y_grid - cy) * z / fy

                #flip x and y
                x = np.flip(x, 0)
                y = np.flip(y, 0)

                # Stack to get 3D points in the camera coordinate system
                points_3d = np.stack((x / x.max(), y / y.max(), z), axis=-1).reshape(-1, 3)
                # Ensure 'croppedimage' is in RGB format. If it's in BGR (common with OpenCV), convert it:
                croppedimage_rgb = cv2.cvtColor(orgframe, cv2.COLOR_BGR2RGB)

                # Flatten the RGB frame to match the points_3d array shape
                colors = croppedimage_rgb.reshape(-1, 3)

                # Create Open3D point cloud object
                pcd = o3d.geometry.PointCloud()

                # Set points
                pcd.points = o3d.utility.Vector3dVector(points_3d)

                # Set colors
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

                # Visualize the point cloud
                o3d.visualization.draw_geometries([pcd])
                #save the point cloud to a file
                o3d.io.write_point_cloud("pointcloud.ply", pcd)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break