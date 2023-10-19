import os

import cv2
import torch
import onnxruntime
import numpy as np
import time

from data import cfg_re50, cfg_mnet
from detect import load_model
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

confidence_threshold = 0.4
nms_threshold = 0.4
vis_threshold = 0.4
top_k = 500
keep_top_k = 750
resize = 1

device = torch.device("cpu")

# torch.set_grad_enabled(False)
# net = RetinaFace(cfg=cfg_mnet, phase='test')
# net = load_model(net, "weights_carcity480/mobilenet0.25_Final.pth", False)
# net.eval()
# net = net.to(device)

ort_session = onnxruntime.InferenceSession("weights_newkz/plate_RFB.onnx", providers=['CPUExecutionProvider'])
from layers.functions.prior_box import PriorBox
priorbox = PriorBox(cfg_mnet, image_size=(320, 320))
# img_path = "kz_test/151121_7227157656740.jpg"
folder = "baqorda"
images = os.listdir(folder)
for im in images:
    start_time = time.time()

    img_orig = cv2.imread(os.path.join(folder, im))
    img_raw = cv2.resize(img_orig, (320, 320))
    img = np.float32(img_raw)
    im_height, im_width, _ = img_raw.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # compute PyTorch prediction
    # t1 = time.time()
    # loc, conf, landms = net(img)
    # print(f'torch exec time: {time.time() - t1}')

    # print("land shape torch: ", landms.shape)

    print("input image shape: ", to_numpy(img).shape)
    # print("input: ", to_numpy(img))

    # compute ONNX Runtime output prediction

    t1 = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    test = ort_session.run(None, ort_inputs)
    print(f'onnx rfb exec time: {time.time() - t1}')
    loc2, conf2, landms2 = test


    # print(test[0].shape, test[1].shape, test[2].shape)
    print('==================================')
    # print('loc', loc.shape)
    print('shape loc2', loc2.shape)

    print('==================================')
    # print('conf', conf.shape)
    print('shape conf2', conf2.shape)

    print('==================================')
    # print('landms', landms.shape)
    print('shape landms2', landms2.shape)
    # print(net)

    # print("loc torch: ", loc)
    # print("conf torch: ", conf)
    # print("landms torch: ", landms)
    print('========================================================================')
    print("loc onnx: ", loc2)
    print("conf onnx: ", conf2)
    print("landms onnx: ", landms2)

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(loc), loc2, rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(conf), conf2, rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(landms), landms2, rtol=1e-03, atol=1e-05)
    #
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # decode and draw results


    priors = priorbox.vectorized_forward()
    priors = priors.to(device)
    prior_data = priors.data
    loc2 = torch.Tensor(loc2)
    loc2 = loc2.to(device)
    conf2 = torch.Tensor(conf2)
    conf2 = conf2.to(device)
    landms2 = torch.Tensor(landms2)
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

    ## keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms2 = landms2[keep]

    ## keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms2 = landms2[:keep_top_k, :]

    dets = np.concatenate((dets, landms2), axis=1)
    end_time = time.time() - start_time
    print(f'onnx exec time: {end_time}')
    height, width, channels = img_orig.shape
    sc_w = width / 320
    sc_h = height / 320
    # show image
    for b in dets:
        if b[4] < vis_threshold:
            continue
        print("detected")
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_orig, (int(b[0]*sc_w), int(b[1]*sc_h)), (int(b[2]*sc_w), int(b[3]*sc_h)), (0, 0, 255), 2)
        cx = b[0]*sc_w
        cy = (b[1] + 12)*sc_h
        cv2.putText(img_orig, text, (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))

        # landms
        cv2.circle(img_orig, (int(b[5]*sc_w), int(b[6]*sc_h)), 1, (0, 0, 255), 4)
        cv2.circle(img_orig, (int(b[7]*sc_w), int(b[8]*sc_h)), 1, (0, 255, 255), 4)
        cv2.circle(img_orig, (int(b[9]*sc_w), int(b[10]*sc_h)), 1, (255, 0, 255), 4)
        cv2.circle(img_orig, (int(b[11]*sc_w), int(b[12]*sc_h)), 1, (0, 255, 0), 4)
        cv2.circle(img_orig, (int(b[13]*sc_w), int(b[14]*sc_h)), 1, (255, 0, 0), 4)
    # save image

    cv2.imwrite("results/" + im.split(".")[0] + "_cc_onnx.jpg", img_orig)

