import sys
import numpy as np
import tritonclient.grpc as tritongrpcclient
import time
import queue
from functools import partial

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))

url = '10.66.150.3:8001'
img_path = "curve/baq2.jpeg"
triton_client = tritongrpcclient.InferenceServerClient(url=url, verbose=False)

inputs = []
outputs = []
input_name = "INPUT"
output_name_1 = "OUTPUT"
output_name_2 = "561"
output_name_3 = "562"

all_start = time.time()
image_data = np.fromfile(img_path, dtype='uint8')
image_data = np.expand_dims(image_data, axis=0)

inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
outputs.append(tritongrpcclient.InferRequestedOutput(output_name_1))
# outputs.append(tritongrpcclient.InferRequestedOutput(output_name_2))
# outputs.append(tritongrpcclient.InferRequestedOutput(output_name_3))
inputs[0].set_data_from_numpy(image_data)
# results = triton_client.infer(model_name='ensemble_detection',
#                               inputs=inputs,
#                               outputs=outputs)
user_data = UserData()
results = triton_client.async_infer(model_name='ensemble_detection',
                                       inputs=inputs,
                                       callback=partial(completion_callback, user_data),
                                       outputs=outputs)

(results, error) = user_data._completed_requests.get()

out1 = results.as_numpy(output_name_1)
# out2 = results.as_numpy(output_name_2)
# out3 = results.as_numpy(output_name_3)

# print(out1)
# print(out2)
# print(out3)

all_end = time.time()
print("all exec time:", all_end - all_start)

# import torch
# from utils.box_utils import decode, decode_landm
# from utils.nms.py_cpu_nms import py_cpu_nms
# from layers.functions.prior_box import PriorBox
# from data import cfg_re50, cfg_mnet
# import queue
# import albumentations as A
# from PIL import Image, ImageOps
# import string
# from converter import StrLabelConverter
# import cv2
# priorbox = PriorBox(cfg_mnet, image_size=(480, 640))
# confidence_threshold = 0.5
# nms_threshold = 0.4
# vis_threshold = 0.5
# top_k = 500
# keep_top_k = 750
# resize = 1
# img_orig = cv2.imread(img_path)
# img_raw = cv2.resize(img_orig, (640, 480))
# img = np.float32(img_raw)
# im_height, im_width, _ = img_raw.shape
# scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
# img -= (104, 117, 123)
# img = img.transpose((2, 0, 1))
# img = torch.from_numpy(img).unsqueeze(0)
#
# device = torch.device("cpu")
# priors = priorbox.vectorized_forward()
# priors = priors.to(device)
# prior_data = priors.data
# loc2 = torch.tensor(out1)
# loc2 = loc2.to(device)
# conf2 = torch.tensor(out3)
# conf2 = conf2.to(device)
# landms2 = torch.tensor(out2)
# landms2 = landms2.to(device)
#
# scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
# boxes = decode(loc2.data.squeeze(0), prior_data, cfg_mnet['variance'])
# boxes = boxes * scale / 1
# boxes = boxes.cpu().numpy()
#
# scores = conf2.squeeze(0).data.cpu().numpy()[:, 1]
# landms2 = decode_landm(landms2.data.squeeze(0), prior_data, cfg_mnet['variance'])
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
#
# # _____________________________________________recognition_tensorrt_______________________________________________
# # rec_client = tritongrpcclient.InferenceServerClient(url=url, verbose=VERBOSE)
# # rec_model_metadata = rec_client.get_model_metadata(model_name=rec_model_name, model_version=model_version)
# # rec_model_config = rec_client.get_model_config(model_name=rec_model_name, model_version=model_version)
# # rec_input = tritongrpcclient.InferInput(rec_input_name, rec_input_shape, input_dtype)
# # # print("rec model metadata", rec_model_metadata)
# # # print("rec model config", rec_model_config)
# #
# # converter = StrLabelConverter(string.digits + string.ascii_lowercase)
# # transformer = A.Compose([A.NoOp()])
# # img_copy = img_orig.copy()
# for b in dets:
#     if b[4] < vis_threshold:
#         continue
#     text = "{:.4f}".format(b[4])
#     b = list(map(int, b))
#     cv2.rectangle(img_orig, (int(b[0] * sc_w), int(b[1] * sc_h)), (int(b[2] * sc_w), int(b[3] * sc_h)), (0, 0, 255),
#                   2)
#     cx = b[0] * sc_w
#     cy = (b[1] + 12) * sc_h
#     cv2.putText(img_orig, text, (int(cx), int(cy)),
#                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
#     #
#     # landms
#     cv2.circle(img_orig, (int(b[5] * sc_w), int(b[6] * sc_h)), 1, (0, 0, 255), 4)  # lt
#     cv2.circle(img_orig, (int(b[7] * sc_w), int(b[8] * sc_h)), 1, (0, 255, 255), 4)  # rt
#     cv2.circle(img_orig, (int(b[9] * sc_w), int(b[10] * sc_h)), 1, (255, 0, 255), 4)  # center
#     cv2.circle(img_orig, (int(b[11] * sc_w), int(b[12] * sc_h)), 1, (0, 255, 0), 4)  # lb
#     cv2.circle(img_orig, (int(b[13] * sc_w), int(b[14] * sc_h)), 1, (255, 0, 0), 4)  # rb
#
#     tl = np.array([b[5] * sc_w, b[6] * sc_h]).astype(int)
#     bl = np.array([b[11] * sc_w, b[12] * sc_h]).astype(int)
#
#     tr = np.array([b[7] * sc_w, b[8] * sc_h]).astype(int)
#     br = np.array([b[13] * sc_w, b[14] * sc_h]).astype(int)
#
#     w = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)
#
#     h = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)
#
#     plate_coords = np.array([[0, 0], [0, int(h)], [int(w), 0], [int(w), int(h)]], dtype='float32')
#     plate_box = np.array([tl, bl, tr, br], dtype='float32')
#     transformation_matrix = cv2.getPerspectiveTransform(plate_box, plate_coords)
#     lp_img = cv2.warpPerspective(img_orig, transformation_matrix, (int(w), int(h)))
#     # preprocessed_image = preprocess(lp_img, transform=transformer).unsqueeze(0).contiguous()
#     #
#     # # send request
#     # start = time.time()
#     # user_data = UserData()
#     # rec_input.set_data_from_numpy(preprocessed_image.numpy())
#     # output = tritongrpcclient.InferRequestedOutput(rec_output_name[0])
#     # response = rec_client.async_infer(model_name=rec_model_name,
#     #                                   model_version=model_version,
#     #                                   inputs=[rec_input],
#     #                                   callback=partial(completion_callback, user_data),
#     #                                   outputs=[output])
#     # end = time.time()
#     # print("triton recognition grps tensorrt exec:", end - start)
#     # (results, error) = user_data._completed_requests.get()
#     # out = results.as_numpy('output')
#     # out = torch.tensor(out)
#     # predictions = out.permute(1, 0, 2).contiguous()
#     # prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
#     # predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
#     # predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
#     # print("label", predicted_test_labels)
#
# cv2.imwrite("results/" + img_path.split('/')[1].replace('.jpeg', '_trt.jpg'), img_orig)
# cv2.imwrite("results/" + img_path.split('/')[1].replace('.jpeg', '_trt_plate.jpg'), lp_img)

