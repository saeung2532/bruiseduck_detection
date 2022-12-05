import tensorflow as tf
import numpy as np
import requests
import re
# import tensorflow.compat.v1 as tf
from datetime import date
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = './annotations/label_map.pbtxt'
# detection_model = tf.saved_model.load("../workspace/models/saved_model/")
# detection_model = tf.saved_model.load("./saved_model/")
# category_index = label_map_util.create_category_index_from_labelmap(
#     PATH_TO_LABELS, use_display_name=True)

def show_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inferencev3(model, category_index, image_np, min_score_thresh=0.5, truckno=1):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Actual detection.
    output_dict = show_inference_for_single_image(model, image_np)

    # Find whole duck class
    index_classes = np.where(output_dict['detection_classes'] == 7)

    for index, line in enumerate(list(index_classes[0])):
        # print(index, line)
        if(output_dict["detection_scores"][line] >= 0.3).any():
            # print(output_dict["detection_scores"][line])
            output_dict["detection_scores"][line] = np.array([0.99])

    # Visualization of the results of a detection
    final_img = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh,
        line_thickness=2)

    datenow = date.today().strftime("%Y-%m-%d")
    # truckno = 1
    qty = 1
    code = ""
    score = 0
    box = [0, 0, 0, 0 ]
    for index, line in enumerate(output_dict['detection_classes']):
        if output_dict['detection_scores'][index] > min_score_thresh:
            # print(category_index.get(line))
            name = category_index.get(line)["name"]
            
            if category_index.get(line)["name"] == "Whole duck":
                # print(str(category_index.get(line)["name"]) + " " + str(output_dict['detection_scores'][index]) + " " + str(output_dict['detection_boxes'][index]))
                box = output_dict['detection_boxes'][index]
            else:
                code = re.findall("[F][P][0-9][0-9]",name)[0]
                score = round(output_dict['detection_scores'][index] * 100, 2)

            # response = requests.post('http://192.200.9.215:8888/api/v2/back', json={"date": datenow, "truckno": truckno, "itemcode": code, "qty" : qty})
            # response = requests.post('http://localhost:8888/api/v2/back', json={"date": datenow, "truckno": truckno, "itemcode": code, "itemname": name, "qty" : qty})
            # print("Status code: ", response.status_code, response.json())

    return(final_img, code, score, box)
