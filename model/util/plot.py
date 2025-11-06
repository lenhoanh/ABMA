# draw anomaly_score line of frame-level evaluation
# t-SNE
import os
import cv2
import csv
import numpy as np
import shutil

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn as nn

matplotlib.use('Agg')  # Use the Agg backend to avoid Tkinter usage


def get_img_np(img_tensor_gpu):
    """
    param img_tensor_gpu: tensor image (shape of [1, 3, 256, 256]) in range of [-1, 1]
    :return: numpy.ndarray in range of [0, 255]
    """
    img_np = img_tensor_gpu.cpu().data.numpy()  # [1, 3, 256, 256]
    img_np = img_np.squeeze(0)  # shape = (3, 256,256)
    img_np = np.transpose(img_np, (1, 2, 0))  # shape = (256,256,3)
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert images from [-1; 1] to [0; 255]
    return img_np


def get_blended_heatmap(target, predict, blend_weight=0.5):
    """
    Trả về ảnh heat map được blend giữa heat map từ error và ảnh dự đoán.
    
    Parameters:
      - target: tensor có shape [1, 3, H, W] với giá trị trong khoảng [-1, 1]
      - predict: tensor có shape [1, 3, H, W] với giá trị trong khoảng [-1, 1]
      - blend_weight: trọng số của heat map (giá trị từ 0 đến 1). Ví dụ: blend_weight=0.5 sẽ cho blend đều giữa 2 ảnh.
      
    Returns:
      - blended: ảnh kết quả (numpy array, kiểu uint8, định dạng BGR) hiển thị sự kết hợp giữa ảnh dự đoán và heat map của error.
    """
    # Lấy heat map màu từ error bằng hàm đã có
    color_error = get_color_error_fr_blue(target, predict)

    # Chuyển đổi predict từ tensor sang ảnh uint8
    # Đầu tiên, chuyển về CPU, detach, chuyển về numpy và scale về [0, 255]
    predict_img = ((predict[0].cpu().detach() + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)

    # Chuyển từ layout [C, H, W] sang [H, W, C]
    predict_img = np.transpose(predict_img, (1, 2, 0))

    # Chuyển đổi ảnh dự đoán từ RGB sang BGR (OpenCV hiển thị theo BGR)
    predict_img = predict_img[:, :, ::-1]

    # Blend 2 ảnh: Sử dụng blend_weight cho heat map, và (1 - blend_weight) cho ảnh dự đoán
    blended = cv2.addWeighted(predict_img, 1 - blend_weight, color_error, blend_weight, 0)

    return blended


def get_color_error_fr_blue(target, predict):
    """
    param target: with shape of [1, 3, 256, 256] in range of [-1, 1]
    param predict: with shape of [1, 3, 256, 256] in range of [-1, 1]
                   remember: batch_size = 1, if not error will appear
    """
    """
    img_np = img_tensor_gpu.cpu().data.numpy()  # [1, 3, 256, 256]
    img_np = img_np.squeeze(0)  # shape = (3, 256,256)
    img_np = np.transpose(img_np, (1, 2, 0))  # shape = (256,256,3)
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert images from [-1; 1] to [0; 255]
    """
    # mse_error_frame(predict, target)
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((predict[0] + 1) / 2, (target[0] + 1) / 2)  # convert [-1,1] to [0,1],
    # print(f"error.shape = {error.shape}")  # = torch.Size([3, 256, 256])
    error_fr = error[0].cpu().data.detach().numpy()  # convert tensor gpu to cpu, then to numpy
    # print(f"error_fr.shape = {error_fr.shape}")  # = (256, 256)

    error_fr = error_fr[:, :, np.newaxis]

    #error_fr = (error_fr - np.min(error_fr)) / (np.max(error_fr) - np.min(error_fr))  # normalized
    min_val = np.min(error_fr)
    max_val = np.max(error_fr)

    if max_val - min_val == 0:
        error_fr = np.zeros_like(error_fr)
    else:
        error_fr = (error_fr - min_val) / (max_val - min_val)

    error_fr = error_fr * 255  # convert [0,1] to [0,255]
    error_fr = error_fr.astype(dtype=np.uint8)
    color_error_fr = cv2.applyColorMap(error_fr, cv2.COLORMAP_JET)

    # print(f"color_error_fr.shape = {color_error_fr.shape}")  # = (256, 256, 3)
    color_error_fr = color_error_fr.astype(np.uint8)[:, :, [2, 1, 0]]  # change RGB to BGR in Channel
    return color_error_fr


def plot_anomaly_scores_MNAD(idx, frame_ids, scores, starts, ends, file_path):
    """
    Plot the graph of anomaly scores of a video
    """

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(frame_ids, scores, label='Anomaly Scores', color='blue')

    # Add the title
    plt.title(f"Anomaly Scores of Video {idx:02d}", fontsize=14)

    # Add labels
    plt.xlabel("Frame ID", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)

    # Đặt gốc tọa độ bắt đầu từ (0, 0)
    plt.xlim(0, max(frame_ids) + 1)
    plt.ylim(0.0, 1.0)

    # Add ground truth anomaly rectangles
    for rs, re in zip(starts, ends):
        current_axis = plt.gca()
        #current_axis.add_patch(Rectangle((rs, 0), re - rs, np.max(scores) + 0.02, facecolor="pink", alpha=0.5))
        current_axis.add_patch(Rectangle((rs, 0), re - rs, 1.0, facecolor="pink", alpha=0.5))

    # Add legend
    from matplotlib.patches import Patch
    pink_patch = Patch(facecolor='pink', alpha=0.5, label='Ground Truth Anomaly')
    plt.legend(handles=[pink_patch], loc='upper left', fontsize='small')

    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_anomaly_heatmaps_lines_pastfuture(idx, targets, predicts,
                                           frame_ids, scores, starts, ends,
                                           export_dir, dpi=300, dataset='ped2', method='MNAD'):
    """
    Export results in a gif image:
    row 1. 04 images: Ground Truth (Past), Predicted Frame (Past), Error Map (Past), Anomaly Map (Past)
    row 2. 04 images: Ground Truth (Future), Predicted Frame (Future), Error Map (Future), Anomaly Map (Future)
    row 3. A graph of Anomaly Scores
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    # Đảm bảo thư mục export tồn tại
    anomaly_score_dir = os.path.join(export_dir, 'anomaly_score')
    os.makedirs(anomaly_score_dir, exist_ok=True)

    # Khoảng cách giữa các mốc trên trục hoành
    xticks_step = {
        'ped2': 10,
        'avenue': 50,
        'shanghaitech': 20,
        'iitb': 20
    }
    num_frames = len(frame_ids)

    for i in range(num_frames):
        # Xử lý ảnh quá khứ
        past_gt = get_img_np(targets[i]["past"])
        past_pred = get_img_np(predicts[i]["past"])
        past_err = get_color_error_fr_blue(targets[i]["past"], predicts[i]["past"])
        past_heat = get_blended_heatmap(targets[i]["past"], predicts[i]["past"])

        # Xử lý ảnh tương lai
        future_gt = get_img_np(targets[i]["future"])
        future_pred = get_img_np(predicts[i]["future"])
        future_err = get_color_error_fr_blue(targets[i]["future"], predicts[i]["future"])
        future_heat = get_blended_heatmap(targets[i]["future"], predicts[i]["future"])

        # Tạo figure với GridSpec: 3 hàng, 4 cột
        fig = plt.figure(figsize=(16, 14))  # (20, 16)
        gs = GridSpec(3, 4, height_ratios=[1, 1, 1.5])

        past_titles = ['Ground Truth (Past)', 'Predicted Frame (Past)', 'Error Map (Past)', 'Anomaly Map (Past)']
        future_titles = ['Ground Truth (Future)', 'Predicted Frame (Future)', 'Error Map (Future)',
                         'Anomaly Map (Future)']

        # Hàng 1: ảnh quá khứ
        for j, (img, title) in enumerate(zip([past_gt, past_pred, past_err, past_heat], past_titles)):
            ax = fig.add_subplot(gs[0, j])
            ax.set_title(title, fontsize=10)
            ax.imshow(img)
            ax.axis('off')

        # Hàng 2: ảnh tương lai
        for j, (img, title) in enumerate(zip([future_gt, future_pred, future_err, future_heat], future_titles)):
            ax = fig.add_subplot(gs[1, j])
            ax.set_title(title, fontsize=10)
            ax.imshow(img)
            ax.axis('off')

        # Hàng 3: biểu đồ Anomaly Scores
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(frame_ids[:i + 1], scores[:i + 1], linestyle='-', lw=2, label=method, color='blue')

        ax5.set_xlim(0, max(frame_ids) + 1)
        ax5.set_ylim(0.0, 1.0)
        ax5.set_xlabel('Frame ID', fontsize=12)
        ax5.set_ylabel('Anomaly Score', fontsize=12)
        ax5.set_title(f'Anomaly Scores of Video {idx:02d}', fontsize=12)

        # step = xticks_step[dataset]
        if "ped2" in dataset:
            step = xticks_step["ped2"]
        elif "avenue" in dataset:
            step = xticks_step["avenue"]
        elif "shanghaitech" in dataset:
            step = xticks_step["shanghaitech"]
        elif "iitb" in dataset:
            step = xticks_step["iitb"]
        else:
            step = xticks_step["ped2"]

        ax5.set_xticks(np.arange(0, max(frame_ids) + 1, step))
        ax5.tick_params(axis='both', which='major', labelsize='x-small')

        # Thêm vùng GT anomaly
        for rs, re in zip(starts, ends):
            ax5.add_patch(plt.Rectangle((rs, 0), re - rs, 1.0, facecolor="pink", alpha=0.5))

        # Thêm legend
        pink_patch = Patch(facecolor='pink', alpha=0.5, label='Ground Truth Anomaly')
        handles, labels = ax5.get_legend_handles_labels()
        handles.append(pink_patch)
        labels.append('Ground Truth Anomaly')
        ax5.legend(handles=handles, labels=labels, loc='upper left', fontsize='small')

        # Lưu hình ảnh
        graph_image_path = os.path.join(anomaly_score_dir, f'frame_{idx:02d}_{i:04d}.png')
        plt.savefig(graph_image_path, dpi=dpi)
        plt.close()

    # Chuyển các ảnh thành video
    output_video_path = os.path.join(export_dir, f'video_{idx:02d}.mp4')
    images_to_video(image_folder=anomaly_score_dir, output_video=output_video_path, fps=20)


def images_to_video(image_folder, output_video, fps=30, delete_folder=True):
    """
    # Example usage
        images_to_video(image_folder='/path/to/images', output_video='output.mp4', fps=30)
    """
    # Get the list of image filenames in the folder
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                          file.endswith(('png', 'jpg', 'jpeg'))])

    # Load the first image to get dimensions
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on file extension (e.g., 'XVID' for AVI)
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through each image and write to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Release the VideoWriter object
    out.release()
    print(f"Video = {output_video} created successfully!")

    """
    # Delete the image files in the folder
    if delete_folder:
        for image_file in image_files:
            os.remove(image_file)
        print("Image files deleted successfully!")
    """

    if delete_folder:
        # Delete the folder and all its files
        shutil.rmtree(image_folder)
        print(f"Folder = {image_folder} and all its files deleted successfully!")


class MetricsResult(object):
    def __init__(self):
        self.alpha = 0.0
        self.auc = 0.0
        self.eer = 0.0
        self.accuracy = 0.0
        self.f1_score = 0.0
        """
        Trong bài toán phát hiện bất thường, vì lớp bất thường là lớp quan trọng cần được dự đoán chính xác, 
        nên trong một số trường hợp ta có thể chấp nhận báo động nhầm bất thường còn hơn bỏ sót bất thường, 
        tức là ta chấp nhận FPR cao để đạt được FNR thấp.

        Tuy nhiên, nếu tỷ lệ báo động nhầm quá cao thì hệ thống bị đánh giá là không có độ tin cậy cao, 
        nên trong một số trường hợp, tỷ lệ báo động nhầm FAR có thể được sử dụng khi so sánh giữa các phương pháp.
        """

        # True Positive Rate (TPR): dự đoán là Positive (bất thường) và thực tế cũng là Positive (bất thường)
        self.tpr = []
        # False Negative Rate (FNR) = 1 - TPR: dự đoán là Negative (bình thường) và thực tế là Positive (bất thường)
        # False Negative Rate (FNR) còn được gọi là Miss Detection Rate (MDR) là tỉ lệ bỏ sót điểm thực sự bất thường.
        # self.fnr = []

        # False Positive Rate (FPR): dự đoán là Positive (bất thường) nhưng thực tế là Negative (bình thường) 
        # False Positive Rate (FPR) còn được gọi là False Alarm Rate (FAR) là tỉ lệ báo động nhầm
        self.fpr = []
        # True Negative Rate (TNR) = 1 - FPR: dự đoán là Negative (bình thường) nhưng thực tế là Negative (bình thường)
        # self.tnr = [] 

        self.idx = 0  # optimal_idx
        self.threshold = 0.0  # optimal_threshold
        self.labels = []
        self.scores = []

    def reset(self):
        self.alpha = 0.0
        self.auc = 0.0
        self.eer = 0.0
        self.accuracy = 0.0
        self.f1_score = 0.0

        self.tpr = []
        # self.fnr = []

        self.fpr = []
        # self.tnr = [] 

        self.idx = 0  # optimal_idx
        self.threshold = 0.0  # optimal_threshold
        self.labels = []
        self.scores = []

    def update(self, alpha=0.0, auc=0.0, eer=0.0, accuracy=0.0, f1_score=0.0,
               fpr=None, tpr=None, idx=0, threshold=0.0, labels=None, scores=None):
        if labels is None:
            labels = []
        if scores is None:
            scores = []

        if tpr is None:
            tpr = []

        if fpr is None:
            fpr = []

        self.alpha = alpha
        self.auc = auc
        self.eer = eer
        self.accuracy = accuracy
        self.f1_score = f1_score

        self.tpr = tpr
        # self.fnr = 1.0 - tpr

        self.fpr = fpr
        # self.tnr = 1.0 - fpr

        self.idx = idx
        self.threshold = threshold
        self.labels = labels
        self.scores = scores


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def print_screen(logger, key, val, format_string=None):
    """
    In giá trị của biến với định dạng cho trước.
        :param logger: logger for printing
        :param key: Một chuỗi (string) đại diện cho tên hoặc loại thông tin bạn đang ghi.
        :param val: Danh sách chứa các giá trị (1 hoặc nhiều), sử dụng cho định dạng.
        :param format_string: Chuỗi định dạng cho các giá trị

    # Ví dụ sử dụng hàm
    x = 42
    print_screen('x', x, 'Giá trị của x là: {}')

    y = 3.14159
    print_screen('y', y, 'Giá trị của y là: {:.2f}')  # Chỉ hiển thị 2 chữ số thập phân
    """
    if format_string is not None:
        formatted_val = format_string.format(*val)
        print(f'{key}: {formatted_val}')
        logger.info(f'{key}: {formatted_val}')
    else:
        print(f'{key} : {val}')
        logger.info(f'{key} : {val}')


def save_alpha_auc_values(alpha_values, auc_values, filename='alpha_auc.csv'):
    """Save alpha and AUC values to a CSV file using the csv module with formatted output."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Alpha', 'AUC'])  # Write header
        for alpha, auc in zip(alpha_values, auc_values):
            # Format alpha and auc to two decimal places
            writer.writerow([f'{alpha:.2f}', f'{auc:.2f}'])


def save_metrics(metrics_result, file_path):
    """Save metrics_result to a CSV file."""
    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Ghi tiêu đề
        writer.writerow(
            ['alpha', 'auc', 'eer', 'accuracy', 'f1_score', 'fpr', 'tpr', 'idx', 'threshold', 'labels', 'scores'])
        # Ghi các giá trị
        writer.writerow([
            metrics_result.alpha,
            metrics_result.auc,
            metrics_result.eer,
            metrics_result.accuracy,
            metrics_result.f1_score,
            ','.join(map(str, metrics_result.fpr)),  # chuyển fpr thành chuỗi
            ','.join(map(str, metrics_result.tpr)),  # chuyển tpr thành chuỗi
            metrics_result.idx,
            metrics_result.threshold,
            ','.join(map(str, metrics_result.labels)),  # chuyển labels thành chuỗi
            ','.join(map(str, metrics_result.scores))  # chuyển scores thành chuỗi
        ])


def minmax_normalization(x, max_val, min_val, eps=1e-6):
    denom = max_val - min_val
    if denom < eps:
        return np.zeros_like(x)  # hoặc x - min_val
    return (x - min_val) / denom
