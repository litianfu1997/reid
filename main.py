from detector import Detector
import cv2
import argparse
import glob
import os
import sys

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn
from sklearn import preprocessing

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from demo.predictor import FeatureExtractionDemo

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default='logs/market1501/bagtricks_R50/config.yaml',
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=['out/person_1.jpg'],
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'logs/market1501/bagtricks_R50/model_best.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


# 归一化
def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


# 检测同一个目标
def detector_fun(video_path, input_feat, save=False):
    # 实例化目标检测模型
    detector = Detector()
    # 读取视频流
    cap = cv2.VideoCapture(video_path)

    # 保存视频
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编解码器
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        out = cv2.VideoWriter('result.avi', fourcc, fps, (1280, 720))  # 写入视频

    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
        class_list = ['person']
        if ret:
            bboxes = detector.detect(frame, class_list)
            bboxes_temp = []
            out_feats = []
            if len(bboxes) > 0:
                count = 0
                for x1, y1, x2, y2, lbl, conf in bboxes:
                    count += 1
                    color = (0, 255, 0)
                    # 裁剪各个行人
                    ximg = frame[y1:y2, x1:x2]
                    # 对每个行人图像进行特征编码
                    out_feat = demo.run_on_image(ximg)
                    out_feat = postprocess(out_feat)
                    # 保存每个行人的特征编码
                    out_feats.append(out_feat)
                    # 保存行人框
                    bboxes_temp.append((x1, y1, x2, y2))

                dist_array = []
                # 计算各个行人的距离，可以修改为矩阵运算
                for ofeat in out_feats:
                    dist = np.linalg.norm(ofeat - input_feat)
                    dist_array.append(dist)
                # 距离最小的下标
                argmin = np.argmin(dist_array)
                # 获取目标框
                array_ = bboxes_temp[argmin]
                x1, y1, x2, y2 = array_

                # 只显示距离小于0.8的
                if dist_array[argmin] <= 0.8:
                    print("距离：" + str(dist_array[argmin]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
                    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    if save:
                        out.write(frame)  # 写入帧
                    cv2.imshow("img", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                    if save:
                        out.write(frame)  # 写入帧
                    cv2.imshow("img", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 获取目标图片特征
def target_feat(img_path):
    # 对目标图片进行特征编码
    if img_path:
        if PathManager.isdir(img_path):
            img_path = glob.glob(os.path.expanduser(img_path))
            assert img_path, "The input path(s) was not found"

        img = cv2.imread(img_path)
        feat = demo.run_on_image(img)
        feat = postprocess(feat)
        # 保存目标图片的特征编码
        # np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)
        return feat


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    PathManager.mkdirs(args.output)
    # 目标图片
    input_feat = target_feat('out/person_1.jpg')

    # 对视频进行行人重识别
    detector_fun('img/TownCentreXVID.avi', input_feat=input_feat, save=False)
