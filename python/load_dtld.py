from __future__ import print_function

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

import cv2
from dtld_parsing.calibration import CalibrationData
from dtld_parsing.driveu_dataset import DriveuDatabase
from dtld_parsing.three_dimensional_position import ThreeDimensionalPosition


__author__ = "Andreas Fregin, Julian Mueller and Klaus Dietmayer"
__maintainer__ = "Julian Mueller"
__email__ = "julian.mu.mueller@daimler.com"


np.set_printoptions(suppress=True)


# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", help="DTLD label files (.json)", type=str, required=True)
    parser.add_argument("--calib_dir", help="calibration directory where .yml are stored", type=str, required=True)
    parser.add_argument(
        "--data_base_dir",
        default="",
        help="only use this if the image file paths in the"
        "label files are not up to date. Do NOT change the"
        "internal DTLD folder structure!",
        type=str,
    )
    return parser.parse_args()


def main(args):

    # Load database
    database = DriveuDatabase(args.label_file)
    if not database.open(args.data_base_dir):
        return False

    # Load calibration
    calibration_left = CalibrationData()
    intrinsic_left = calibration_left.load_intrinsic_matrix(args.calib_dir + "/intrinsic_left.yml")
    rectification_left = calibration_left.load_rectification_matrix(args.calib_dir + "/rectification_left.yml")
    projection_left = calibration_left.load_projection_matrix(args.calib_dir + "/projection_left.yml")
    extrinsic = calibration_left.load_extrinsic_matrix(args.calib_dir + "/extrinsic.yml")
    distortion_left = calibration_left.load_distortion_matrix(args.calib_dir + "/distortion_left.yml")

    calibration_right = CalibrationData()
    projection_right = calibration_right.load_projection_matrix(args.calib_dir + "/projection_right.yml")

    threed_position = ThreeDimensionalPosition(
        calibration_left=calibration_left,
        calibration_right=calibration_right,
        binning_x=2,
        binning_y=2,
        roi_offset_x=0,
        roi_offset_y=0,
    )

    logging.info("Intrinsic Matrix:\n\n{}\n".format(intrinsic_left))
    logging.info("Extrinsic Matrix:\n\n{}\n".format(extrinsic))
    logging.info("Projection Matrix:\n\n{}\n".format(projection_left))
    logging.info("Rectification Matrix:\n\n{}\n".format(rectification_left))
    logging.info("Distortion Matrix:\n\n{}\n".format(distortion_left))

    # create axes
    ax1 = plt.subplot(111)

    # Visualize image by image
    for idx_d, img in enumerate(database.images):

        # Get disparity image
        img_disp = img.visualize_disparity_image()
        rects = img.map_labels_to_disparity_image(calibration_left)

        disparity_image = img.get_disparity_image()

        # Plot labels into disparity image
        for rect in rects:
            cv2.rectangle(
                img_disp,
                (int(rect[0]), int(rect[1])),
                (int(rect[0]) + int(rect[2]), int(rect[1]) + int(rect[3])),
                (255, 255, 255),
                2,
            )
        # Get color image with labels
        img_color = img.get_labeled_image()
        img_color = cv2.resize(img_color, (1024, 440))

        # Demo how to get 3D position
        for o in img.objects:
            # Camera Coordinate System: X right, Y down, Z front
            threed_pos = threed_position.determine_three_dimensional_position(
                o.x, o.y, o.width, o.height, disparity_image=disparity_image
            )
            # Get in vehicle coordinates: X front, Y left and Z up
            threed_pos_numpy = np.array([threed_pos.get_pos()[0], threed_pos.get_pos()[1], threed_pos.get_pos()[2], 1])
            threed_pos_vehicle_coordinates = extrinsic.dot(threed_pos_numpy)
            print("TL position (rear axle): ", threed_pos_vehicle_coordinates)
        # Plot side by side
        img_concat = np.concatenate((img_color, img_disp), axis=1)
        # Because of the weird qt error in gui methods in opencv-python >= 3
        # imshow does not work in some cases. You can try it by yourself.
        # cv2.imshow("DTLD_visualized", img_concat)
        # cv2.waitKey(1)
        img_concat_rgb = img_concat[..., ::-1]
        if idx_d == 0:
            im1 = ax1.imshow(img_concat_rgb)
        plt.ion()
        im1.set_data(img_concat_rgb)
        plt.pause(0.001)
        plt.draw()


if __name__ == "__main__":
    main(parse_args())
