from modules.utils import visualization_utils as vis_util
from modules.utils import ops as utils_ops
from modules.utils import label_map_util
from modules.utils.helpers import should_consider_image
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import pytz
import datetime


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

MAX_BOXES_TO_DRAW = 2
THRESHOLD = 0.97
AREA_THRESHOLD = 0.03
TZ = pytz.timezone('Asia/Singapore')
DATETIME_FORMAT = '%Y%m%d-%H%M'


def scale_down(image, percentage=60):
    width = int(image.shape[1] * percentage / 100)
    height = int(image.shape[0] * percentage / 100)
    dim = (width, height)
    return cv2.resize(image, dim)


def convert_to_rgba(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def determine_section(xmin, xmax):
    midpoint = (xmin + xmax) / 2
    if midpoint <= 0.33:
        return 0
    if 0.33 < midpoint < 0.66:
        return 1
    else:
        return 2


model_name = 'modules/nn/inference_graph'
train_dir = 'modules/nn/training'
path_to_frozen_graph = model_name + '/frozen_inference_graph.pb'
path_to_labels = f"{train_dir}/labelmap.pbtxt"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(
    path_to_labels, use_display_name=True
)


def run_inference_for_single_image(image, graph, tensor_dict, sess):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(
            detection_masks, [0, 0, 0], [real_num_detection, -1, -1]
        )
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8
        )
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
        np.uint8
    )
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


IMAGES_TAKEN = {}


def recognize(image, target='', debug='', meta={}):

    os.mkdir(target) if not os.path.exists(target) else None
    os.mkdir(debug) if not os.path.exists(debug) else None
    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes',
                'detection_masks',
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name
                    )

            image_np = np.array(Image.open(image))
            np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(
                image_np, detection_graph, tensor_dict, sess,
            )
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=4,
                max_boxes_to_draw=MAX_BOXES_TO_DRAW,
                min_score_thresh=THRESHOLD,
                area_thresh=AREA_THRESHOLD,
            )
            detections = []
            found = False
            for i in range(MAX_BOXES_TO_DRAW):
                score = output_dict['detection_scores'][i]
                ymin, xmin, ymax, xmax = [
                    float(c) for c in output_dict['detection_boxes'][i]
                ]
                area = (ymax - ymin) * (xmax - xmin)
                class_id = output_dict['detection_classes'][i]

                section = determine_section(xmin, xmax)
                consider = should_consider_image(section=section, meta=meta)

                if score > THRESHOLD and area > AREA_THRESHOLD:
                    print("Section: ", section)
                    print("Meta:", meta)
                    if not consider:
                        print(
                            f"Skipping image because this image should not be considered"
                        )
                        print(f"Meta: {meta}")
                        print(f"Section: {section}")
                        print(f"image id: {class_id}")
                        print(f"Image filename: {image.filename}")
                        continue
                    if str(class_id) in IMAGES_TAKEN:
                        print(f'Skipping duplicate for image id {class_id}')
                        continue
                    found = True
                    IMAGES_TAKEN[str(class_id)] = True
                    print(
                        f"Found image id {category_index[class_id]['name']} - score: {score}"
                    )
                    print(
                        f"Coordinates: ymin, xmin, ymax, xmax = ({ymin}, {xmin}, {ymax}, {xmax}) "
                    )
                    print(f"Area: {area}")
                    detections.append(
                        {
                            'id': int(class_id),
                            'name': category_index[class_id]['name'],
                            'score': float(output_dict['detection_scores'][i]),
                            'ymin': ymin,
                            'xmin': xmin,
                            'ymax': ymax,
                            'xmax': xmax,
                            'section': section,
                        }
                    )
            rgb_image = convert_to_rgba(image_np)
            now = datetime.datetime.now(TZ).strftime(DATETIME_FORMAT)
            filename = f'{now}_{image.filename}'
            if found:
                cv2.imwrite(os.path.join(target, filename), rgb_image)
            else:
                print('Not found any known image', filename)
            if debug:
                cv2.imwrite(os.path.join(debug, filename), rgb_image)
            return detections
