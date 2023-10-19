import cv2
import torch
import onnxruntime
import numpy as np
import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = "kz_test/test_image.png"

ort_session = onnxruntime.InferenceSession("weights_newkz/PlateDetector_mobilenet640x480.onnx", providers=['CPUExecutionProvider'])
start_time = time.time()
img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_raw = cv2.resize(img_orig, (640, 480))
img = np.float32(img_raw)
im_height, im_width, _ = img_raw.shape
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
test = ort_session.run(None, ort_inputs)
loc, conf, landms = test
end_time = time.time() - start_time

print(f'onnx exec time: {end_time}')
print(loc)
# device = torch.device("cpu")
# from layers.functions.prior_box import PriorBox
# from data import cfg_mnet
# from utils.box_utils import decode, decode_landm
# from utils.nms.py_cpu_nms import py_cpu_nms
#
# confidence_threshold = 0.7
# nms_threshold = 0.4
# vis_threshold = 0.7
# top_k = 5000
# keep_top_k = 750
# resize = 1
#
# priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
# priors = priorbox.forward()
# priors = priors.to(device)
# prior_data = priors.data
# loc = torch.Tensor(loc)
# loc = loc.to(device)
# conf2 = torch.Tensor(conf)
# conf2 = conf2.to(device)
# landms = torch.Tensor(landms)
# landms = landms.to(device)
#
# boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
# boxes = boxes * scale / 1
# boxes = boxes.cpu().numpy()
#
# scores = conf2.squeeze(0).data.cpu().numpy()[:, 1]
# landms2 = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
# scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                        img.shape[3], img.shape[2]])
# scale1 = scale1.to(device)
# landms2 = landms2 * scale1 / 1
# landms2 = landms2.cpu().numpy()
# ## ignore low scores
# inds = np.where(scores > confidence_threshold)[0]
# boxes = boxes[inds]
# landms2 = landms2[inds]
# scores = scores[inds]
# ##
# ## keep top-K before NMS
# order = scores.argsort()[::-1][:top_k]
# boxes = boxes[order]
# landms2 = landms2[order]
# scores = scores[order]
#
# ## do NMS
# dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
# keep = py_cpu_nms(dets, nms_threshold)
#
# ## keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
# dets = dets[keep, :]
# landms2 = landms2[keep]
#
# ## keep top-K faster NMS
# dets = dets[:keep_top_k, :]
# landms2 = landms2[:keep_top_k, :]
#
# dets = np.concatenate((dets, landms2), axis=1)
#
# height, width, channels = img_orig.shape
# sc_w = width / 640
# sc_h = height / 480
# # show image
# for b in dets:
#     if b[4] < vis_threshold:
#         continue
#     text = "{:.4f}".format(b[4])
#     b = list(map(int, b))
#     cv2.rectangle(img_orig, (int(b[0]*sc_w), int(b[1]*sc_h)), (int(b[2]*sc_w), int(b[3]*sc_h)), (0, 0, 255), 2)
#     cx = b[0]*sc_w
#     cy = (b[1] + 12)*sc_h
#     cv2.putText(img_orig, text, (int(cx), int(cy)),
#                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
#
#     # landms
#     cv2.circle(img_orig, (int(b[5]*sc_w), int(b[6]*sc_h)), 1, (0, 0, 255), 4)
#     cv2.circle(img_orig, (int(b[7]*sc_w), int(b[8]*sc_h)), 1, (0, 255, 255), 4)
#     cv2.circle(img_orig, (int(b[9]*sc_w), int(b[10]*sc_h)), 1, (255, 0, 255), 4)
#     cv2.circle(img_orig, (int(b[11]*sc_w), int(b[12]*sc_h)), 1, (0, 255, 0), 4)
#     cv2.circle(img_orig, (int(b[13]*sc_w), int(b[14]*sc_h)), 1, (255, 0, 0), 4)
# # save image
#
# cv2.imwrite("results/" + img_path.split("/")[1].split(".")[0] + "_onnx.jpg", img_orig)