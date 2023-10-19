import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
from functools import partial
import cv2
import numpy as np
import torch
import time
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from data import cfg_re50, cfg_mnet
import queue
import albumentations as A
from PIL import Image, ImageOps
import string
from converter import StrLabelConverter
import asyncio
import tritonclient.grpc.aio as grpcclient

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def preprocess(img, transform=None, imwrite=None):
    original_image = img.copy()
    h, w, _ = img.shape
    left = 0
    right = 0
    top = 0
    bottom = 0
    if w > h:
        h = int(h * 128/ w)
        w = 128
        top = (32 - h) // 2
        bottom = 32 - h - top
    else:
        w = int(w * 32 / h)
        h = 32
        left = (128 - w) // 2
        right = 128 - w - left
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        im_pil = Image.fromarray(img)
        img = resize_with_padding(im_pil, (128, 32))
        x = np.asarray(img)
        if transform is not None:
            x = transform(image=x)['image']
    else:
        x = cv2.resize(img, (w, h))
        if transform is not None:
            x = transform(image=x)['image']
        x = cv2.copyMakeBorder(x, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    x = x.astype(np.float32) / 255.
    x = x.transpose(2, 0, 1)
    x = torch.tensor(x)
    return x

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))

priorbox = PriorBox(cfg_mnet, image_size=(480, 640))
confidence_threshold = 0.7
nms_threshold = 0.5
vis_threshold = 0.7
top_k = 500
keep_top_k = 750
resize = 1

all_start = time.time()
im_path = "baq8.jpeg"
img_orig = cv2.imread('curve/' + im_path)
img_raw = cv2.resize(img_orig, (640, 480))
img = np.float32(img_raw)
im_height, im_width, _ = img_raw.shape
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0)

VERBOSE = False
det_input_name = 'input0'
rec_input_name = 'actual_input'
det_input_shape = (1, 3, 480, 640)
rec_input_shape = (1, 3, 32, 128)
input_dtype = 'FP32'
det_output_name = ['output0', '561', '562']
rec_output_name = ['output']
det_model_name = 'retina_tensorrt'
rec_model_name = 'recognition_tensorrt'
url = '10.66.150.3:8001'
model_version = '1'

det_client = tritongrpcclient.InferenceServerClient(url=url, verbose=VERBOSE)
det_model_metadata = det_client.get_model_metadata(model_name=det_model_name, model_version=model_version)
det_model_config = det_client.get_model_config(model_name=det_model_name, model_version=model_version)

# print("det model metadata", model_metadata)
# print("det model config", model_config)

input0 = tritongrpcclient.InferInput(det_input_name, det_input_shape, input_dtype)
input0.set_data_from_numpy(img.numpy())

start = time.time()
output1 = tritongrpcclient.InferRequestedOutput(det_output_name[0])
output2 = tritongrpcclient.InferRequestedOutput(det_output_name[1])
output3 = tritongrpcclient.InferRequestedOutput(det_output_name[2])
# user_data = []
user_data = UserData()
response = det_client.async_infer(model_name=det_model_name,
                                       model_version=model_version,
                                       inputs=[input0],
                                       callback=partial(completion_callback, user_data),
                                       outputs=[output1, output2, output3])
end = time.time()
(results, error) = user_data._completed_requests.get()
print("triton detection grps tensorrt exec:", end - start)
out1 = results.as_numpy('output0')
out2 = results.as_numpy('561')
out3 = results.as_numpy('562')

# postprocess
device = torch.device("cpu")
priors = priorbox.vectorized_forward()
priors = priors.to(device)
prior_data = priors.data
loc2 = torch.tensor(out1)
loc2 = loc2.to(device)
conf2 = torch.tensor(out3)
conf2 = conf2.to(device)
landms2 = torch.tensor(out2)
landms2 = landms2.to(device)

boxes = decode(loc2.data.squeeze(0), prior_data, cfg_mnet['variance'])
boxes = boxes * scale / 1
boxes = boxes.cpu().numpy()

scores = conf2.squeeze(0).data.cpu().numpy()[:, 1]
landms2 = decode_landm(landms2.data.squeeze(0), prior_data, cfg_mnet['variance'])
scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                       img.shape[3], img.shape[2]])
scale1 = scale1.to(device)
landms2 = landms2 * scale1 / 1
landms2 = landms2.cpu().numpy()
## ignore low scores
inds = np.where(scores > confidence_threshold)[0]
boxes = boxes[inds]
landms2 = landms2[inds]
scores = scores[inds]
##
## keep top-K before NMS
order = scores.argsort()[::-1][:top_k]
boxes = boxes[order]
landms2 = landms2[order]
scores = scores[order]

## do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = py_cpu_nms(dets, nms_threshold)

dets = dets[keep, :]
landms2 = landms2[keep]

## keep top-K faster NMS
dets = dets[:keep_top_k, :]
landms2 = landms2[:keep_top_k, :]

dets = np.concatenate((dets, landms2), axis=1)

height, width, channels = img_orig.shape
sc_w = width / 640
sc_h = height / 480

#_____________________________________________recognition_tensorrt_______________________________________________
rec_client = tritongrpcclient.InferenceServerClient(url=url, verbose=VERBOSE)
rec_model_metadata = rec_client.get_model_metadata(model_name=rec_model_name, model_version=model_version)
rec_model_config = rec_client.get_model_config(model_name=rec_model_name, model_version=model_version)
rec_input = tritongrpcclient.InferInput(rec_input_name, rec_input_shape, input_dtype)
# print("rec model metadata", rec_model_metadata)
# print("rec model config", rec_model_config)

converter = StrLabelConverter(string.digits + string.ascii_lowercase)
transformer = A.Compose([A.NoOp()])
img_copy = img_orig.copy()
for b in dets:
    if b[4] < vis_threshold:
        continue
    text = "{:.4f}".format(b[4])
    b = list(map(int, b))
    cv2.rectangle(img_copy, (int(b[0]*sc_w), int(b[1]*sc_h)), (int(b[2]*sc_w), int(b[3]*sc_h)), (0, 0, 255), 2)
    cx = b[0]*sc_w
    cy = (b[1] + 12)*sc_h
    cv2.putText(img_copy, text, (int(cx), int(cy)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
    #
    # landms
    cv2.circle(img_copy, (int(b[5]*sc_w), int(b[6]*sc_h)), 1, (0, 0, 255), 4) # lt
    cv2.circle(img_copy, (int(b[7]*sc_w), int(b[8]*sc_h)), 1, (0, 255, 255), 4) # rt
    cv2.circle(img_copy, (int(b[9]*sc_w), int(b[10]*sc_h)), 1, (255, 0, 255), 4) # center
    cv2.circle(img_copy, (int(b[11]*sc_w), int(b[12]*sc_h)), 1, (0, 255, 0), 4) # lb
    cv2.circle(img_copy, (int(b[13]*sc_w), int(b[14]*sc_h)), 1, (255, 0, 0), 4) # rb

    tl = np.array([b[5]*sc_w, b[6]*sc_h]).astype(int)
    bl = np.array([b[11]*sc_w, b[12]*sc_h]).astype(int)

    tr = np.array([b[7]*sc_w, b[8]*sc_h]).astype(int)
    br = np.array([b[13]*sc_w, b[14]*sc_h]).astype(int)

    w = int(((tr[0] - tl[0]) + (br[0] - bl[0])) / 2)

    h = int(((bl[1] - tl[1]) + (br[1] - tr[1])) / 2)

    plate_coords = np.array([[0, 0], [0, int(h)], [int(w), 0], [int(w), int(h)]], dtype='float32')
    plate_box = np.array([tl, bl, tr, br], dtype='float32')
    transformation_matrix = cv2.getPerspectiveTransform(plate_box, plate_coords)
    lp_img = cv2.warpPerspective(img_orig, transformation_matrix, (int(w), int(h)))
    preprocessed_image = preprocess(lp_img, transform=transformer).unsqueeze(0).contiguous()

    # send request
    start = time.time()
    user_data = UserData()
    rec_input.set_data_from_numpy(preprocessed_image.numpy())
    output = tritongrpcclient.InferRequestedOutput(rec_output_name[0])
    response = rec_client.async_infer(model_name=rec_model_name,
                                      model_version=model_version,
                                      inputs=[rec_input],
                                      callback=partial(completion_callback, user_data),
                                      outputs=[output])
    end = time.time()
    print("triton recognition grps tensorrt exec:", end - start)
    (results, error) = user_data._completed_requests.get()
    out = results.as_numpy('output')
    out = torch.tensor(out)
    predictions = out.permute(1, 0, 2).contiguous()
    prediction_size = torch.IntTensor([predictions.size(0)]).repeat(1)
    predicted_probs, predicted_labels = predictions.detach().cpu().max(2)
    predicted_test_labels = np.array(converter.decode(predicted_labels, prediction_size, raw=False))
    print("label", predicted_test_labels)

all_end = time.time()
print("all exec time:", all_end - all_start)
cv2.imwrite("results/" + im_path.replace('.jpeg', '_triton.jpg'), img_copy)
cv2.imwrite("results/" + im_path.replace('.jpeg', '_triton_plate.jpg'), lp_img)


# ______________________________________________ratina_torch______________________________________________
# VERBOSE = False
# input_name = 'input__0'
# input_shape = (1, 3, 480, 640)
# input_dtype = 'FP32'
# output_name = 'output__0'
# model_name = 'retina_torch'
# url = '10.66.150.3:8001'
# model_version = '1'
#
# torch_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
# model_metadata = torch_client.get_model_metadata(model_name=model_name, model_version=model_version)
# model_config = torch_client.get_model_config(model_name=model_name, model_version=model_version)
# print("model metadata", model_metadata)
# print("model config", model_config)
#
#
# for _ in range(3):
#     input0 = tritonhttpclient.InferInput(input_name, input_shape, input_dtype)
#     input0.set_data_from_numpy(img.numpy(), binary_data=False)
#     start = time.time()
#     output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)
#     response = torch_client.infer(model_name, model_version=model_version,
#                                    inputs=[input0], outputs=[output])
#     end = time.time()
#     print("retina torch exec:", end - start)
#
# out = response.as_numpy('output__0')
# out = np.asarray(out, dtype=np.float32)

# __________________________________________retina onnnx_______________________________________
# VERBOSE = False
# input_name = 'input0'
# input_shape = (1, 3, 480, 640)
# input_dtype = 'FP32'
# output_name = ['output0', '575', '576']
# model_name = 'retina_onnx'
# url = '10.66.150.3:8000'
# model_version = '1'
#
# onnx_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
# model_metadata = onnx_client.get_model_metadata(model_name=model_name, model_version=model_version)
# model_config = onnx_client.get_model_config(model_name=model_name, model_version=model_version)
# print("model metadata", model_metadata)
# print("model config", model_config)
#
# for _ in range(10):
#     input0 = tritonhttpclient.InferInput(input_name, (1, 3, 480, 640), 'FP32')
#     input0.set_data_from_numpy(img.numpy(), binary_data=False)
#
#     start = time.time()
#     output1 = tritonhttpclient.InferRequestedOutput(output_name[0], binary_data=False)
#     output2 = tritonhttpclient.InferRequestedOutput(output_name[1], binary_data=False)
#     output3 = tritonhttpclient.InferRequestedOutput(output_name[2], binary_data=False)
#
#     response = onnx_client.async_infer(model_name, model_version=model_version,
#                                    inputs=[input0], outputs=[output1, output2, output3])
#     end = time.time()
#     print("triton onnx exec:", end - start)
#
# out1 = response.as_numpy('output0')
# out1 = np.asarray(out1, dtype=np.float32)
# out2 = response.as_numpy('575')
# out2 = np.asarray(out2, dtype=np.float32)
# out3 = response.as_numpy('576')
# out3 = np.asarray(out3, dtype=np.float32)
# print(out1.shape, out1)
# print(out2.shape, out2)
# print(out3.shape, out3)

# __________________________________________retina tensorrt_______________________________________
# VERBOSE = False
# input_name = 'input0'
# input_shape = (1, 3, 480, 640)
# input_dtype = 'FP32'
# output_name = ['output0', '561', '562']
# model_name = 'retina_tensorrt'
# url = '10.66.150.3:8000'
# model_version = '1'
#
# tensorrt_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
# model_metadata = tensorrt_client.get_model_metadata(model_name=model_name, model_version=model_version)
# model_config = tensorrt_client.get_model_config(model_name=model_name, model_version=model_version)
# # print("model metadata", model_metadata)
# # print("model config", model_config)
#
# input0 = tritonhttpclient.InferInput(input_name, input_shape, input_dtype)
# input0.set_data_from_numpy(img.numpy())
#
# start = time.time()
# output1 = tritonhttpclient.InferRequestedOutput(output_name[0])
# output2 = tritonhttpclient.InferRequestedOutput(output_name[1])
# output3 = tritonhttpclient.InferRequestedOutput(output_name[2])
# user_data = []
# response = tensorrt_client.async_infer(model_name=model_name,
#                                        model_version=model_version,
#                                        inputs=[input0],
#                                        # callback=partial(callback, user_data),
#                                        outputs=[output1, output2, output3])
# end = time.time()
# print("triton http tensorrt exec:", end - start)
#
# results = response.get_result()
# out1 = results.as_numpy('output0')
# out1 = np.asarray(out1, dtype=np.float32)
# out2 = results.as_numpy('561')
# out2 = np.asarray(out2, dtype=np.float32)
# out3 = results.as_numpy('562')
# out3 = np.asarray(out3, dtype=np.float32)
# print(out1.shape, out1)
# print(out2.shape, out2)
# print(out3.shape, out3)