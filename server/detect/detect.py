import torch
import cv2
import os


def get_analysed_image(results_list, pad_original_image, img_original_shape,
                       color=(56, 56, 255), txt_color=(255, 255, 255)):

    for result in results_list:
        x0, y0, x1, y1 = map(lambda x: int(round(x)),result[:4])
        w, h = 10,4
        outside = y0 - h >= 3
        lw = max(round(sum(pad_original_image.shape) / 2 * 0.003), 2)

        cv2.rectangle(pad_original_image, (x0, y0), (x1, y1),
                      color, 1, lineType=cv2.LINE_AA)

        x1, y1 = x0 + 140, y0 - 12 - 3 if outside else y0 + 12 + 3
        cv2.rectangle(pad_original_image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(pad_original_image,
                    "abnormality: " + str(round(result[-2], 2)),
                    (x0, y0 - 4 if outside else y0 + h + 4),
                    0,
                    lw/4,
                    txt_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)


    return pad_original_image    


def save(img, result_file_path):
    cv2.imwrite(result_file_path,img)


def run(img, result_file_path, pad_original_image=1, img_original_shape=1, weights_relative_path="../weights/best.pt"):

    dir = os.path.dirname(__file__)
    weights_path = os.path.abspath(os.path.join(dir, weights_relative_path))
    try:
        dir_torch = os.path.abspath(os.path.join(dir, ".."))
        torch.hub.set_dir(dir_torch)
        model = torch.hub.load("ultralytics/yolov5", 'custom',
                            path=weights_path)#, force_reload=True)

        results = model(img, size=640)

        analysed_image = get_analysed_image(
            results.xyxy[0].tolist(), pad_original_image, img_original_shape)
        
        save(analysed_image,result_file_path)

        exit_code = 0

    except Exception as e:
        exit_code = 1

    return exit_code