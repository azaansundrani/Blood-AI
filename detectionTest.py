from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("cells/models/detection_model-ex-218--loss-0024.624.h5")
detector.setJsonPath("cells/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="Malaria_1.jpg", output_image_path="imagenew.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

