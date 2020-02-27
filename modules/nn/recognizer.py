from modules.utils import visualization_utils as vis_util
from modules.utils import ops as utils_ops
from modules.utils import label_map_util
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os


MAX_BOXES_TO_DRAW = 5
THRESHOLD = 0.7


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


def recognize(
    image,
    train_dir='modules/nn/training',
    model_dir='modules/nn/inference_graph',
    target=''
):
    model_name = model_dir
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_frozen_graph = model_name + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    path_to_labels = f"{train_dir}/labelmap.pbtxt"

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    def run_inference_for_single_image(image, graph):
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            print(f"Processing {image.filename}...")
            image_np = np.array(Image.open(image))
            np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=THRESHOLD,
            )
            detections = []
            found = False
            for i in range(MAX_BOXES_TO_DRAW):
                if output_dict['detection_scores'][i] > THRESHOLD:
                    found = True
                    class_id = output_dict['detection_classes'][i]
                    print(f"Found image id {category_index[class_id]['name']} - score: {output_dict['detection_scores'][i]}")
                    ymin, xmin, ymax, xmax = [float(c) for c in output_dict['detection_boxes'][i]]
                    section = determine_section(xmin, xmax)
                    print(f"Coordinates: ymin, xmin, ymax, xmax = ({ymin}, {xmin}, {ymax}, {xmax}) ")
                    detections.append({
                        'id': int(class_id),
                        'name': category_index[class_id]['name'],
                        'score': float(output_dict['detection_scores'][i]),
                        'ymin': ymin,
                        'xmin': xmin,
                        'ymax': ymax,
                        'xmax': xmax,
                        'section': section,
                    })
            if found:
                cv2.imwrite(os.path.join(target, image.filename), convert_to_rgba(image_np))
            else:
                print('Not found any known image')
            return detections
