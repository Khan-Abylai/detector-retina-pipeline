import os
#import tensorflow as tf
import numpy as np
import cv2
import time
# import torch
# import onnxruntime
import pycoral.utils.edgetpu as etpu
#print("Tensorflow ", tf.__version__)

confidence_threshold = 0.8
nms_threshold = 0.6
vis_threshold = 0.8
top_k = 5000
keep_top_k = 750
resize = 1

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ------------------------------test tflite model-----------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_path = "uae_test.jpeg"

# Load the TFLite model and allocate tensors
interpreter = etpu.make_interpreter('PlateDetector-mobilenet_float16.tflite')
#interpreter = tf.lite.Interpreter(model_path='PlateDetector-mobilenet_float16.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32
print("floating model: ", floating_model)
input_type = input_details[0]['dtype']
print('input: ', input_type)
output_type = output_details[0]['dtype']
print('output: ', output_type)

height = input_details[0]['shape'][2]
width = input_details[0]['shape'][3]

img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_raw = cv2.resize(img_orig, (width, height))

img = np.float32(img_raw)
# scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose((2, 0, 1))

for _ in range(10):

    input_data = np.expand_dims(img, axis=0).transpose((0, 1, 2, 3))

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.perf_counter()
    interpreter.invoke()
    stop_time = time.perf_counter()

    print(interpreter.get_tensor(output_details[0]['index']).shape)
    print(interpreter.get_tensor(output_details[1]['index']).shape)
    print(interpreter.get_tensor(output_details[2]['index']).shape)

    print('time: {:.4f} ms'.format((stop_time - start_time)*1000))

# ____________decode and draw results____________________
#from layers.functions.prior_box import PriorBox
#from utils.box_utils import decode, decode_landm
#from data import cfg_re50, cfg_mnet
#from utils.nms.py_cpu_nms import py_cpu_nms

# device = torch.device("cuda")

loc2 = interpreter.get_tensor(output_details[0]['index'])
conf2 = interpreter.get_tensor(output_details[2]['index'])
landms2 = interpreter.get_tensor(output_details[1]['index'])

print("shape loc2: ", loc2.shape)
print("shape conf2: ", conf2.shape)
print("shape landms2: ", landms2.shape)
print("loc2: ", loc2)
print("conf2: ", conf2)
print("landms2: ", landms2)

# loc2 = torch.Tensor(loc2)
# loc2 = loc2.to(device)
# conf2 = torch.Tensor(conf2)
# conf2 = conf2.to(device)
# landms2 = torch.Tensor(landms2)
# landms2 = landms2.to(device)
#
# img = torch.from_numpy(img).unsqueeze(0)
# img = img.to(device)
# scale = scale.to(device)
#
# priorbox = PriorBox(cfg_mnet, image_size=(height, width))
# priors = priorbox.forward()
# priors = priors.to(device)
# prior_data = priors.data
#
# boxes = decode(loc2.data.squeeze(0), prior_data, cfg_mnet['variance'])
# boxes = boxes * scale / resize
# boxes = boxes.cpu().numpy()
# scores = conf2.squeeze(0).data.cpu().numpy()[:, 1]
# landms = decode_landm(landms2.data.squeeze(0), prior_data, cfg_mnet['variance'])
# scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                        img.shape[3], img.shape[2]])
# scale1 = scale1.to(device)
# landms = landms * scale1 / resize
# landms = landms.cpu().numpy()
#
# # ignore low scores
# inds = np.where(scores > confidence_threshold)[0]
# boxes = boxes[inds]
# landms = landms[inds]
# scores = scores[inds]
#
# # keep top-K before NMS
# order = scores.argsort()[::-1][:top_k]
# boxes = boxes[order]
# landms = landms[order]
# scores = scores[order]
#
# # do NMS
# dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
# keep = py_cpu_nms(dets, nms_threshold)
# # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
# dets = dets[keep, :]
# landms = landms[keep]
#
# # keep top-K faster NMS
# dets = dets[:keep_top_k, :]
# landms = landms[:keep_top_k, :]
#
# dets = np.concatenate((dets, landms), axis=1)
#
# # show image
# for b in dets:
#     if b[4] < vis_threshold:
#         continue
#     text = "{:.4f}".format(b[4])
#     b = list(map(int, b))
#     cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#     cx = b[0]
#     cy = b[1] + 12
#     cv2.putText(img_raw, text, (cx, cy),
#                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
#     # landms
#     cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#     cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#     cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#     cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
#     cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
#
# cv2.imwrite("results/" + img_path.split("/")[1].split(".")[0] + "_mobilenet_tflite.jpg", img_raw)
