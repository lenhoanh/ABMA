import torch
import numpy as np
import random
import os
import torch.backends.cudnn as cudnn

'''
Để đảm bảo rằng mô hình convAE cho ra kết quả không đổi qua các lần chạy, 
bạn cần thiết lập seed cho tất cả các yếu tố ngẫu nhiên trong quá trình huấn luyện. 
Điều này bao gồm việc cố định các giá trị ngẫu nhiên trong PyTorch, NumPy, và các thư viện ngẫu nhiên khác (nếu có), 
cũng như cấu hình một số tham số cụ thể trong PyTorch để đạt được tính nhất quán.

Dưới đây là các bước để thực hiện điều đó:

- Bước 1: Cố định các seed ngẫu nhiên: Thiết lập seed cho PyTorch, NumPy, và các hàm ngẫu nhiên của Python.
- Bước 2: Thiết lập tùy chọn PyTorch cho tính tái lập: 
    + Điều chỉnh một số cấu hình PyTorch như torch.backends.cudnn.deterministic và torch.backends.cudnn.benchmark.
- Bước 3: Khởi tạo mô hình hoặc biến (nếu cần) với seed cố định: 
    + Ví dụ: Khởi tạo mô hình convAE
    + Khởi tạo biến: self.m_items để đảm bảo rằng self.m_items được khởi tạo với cùng một giá trị qua các lần chạy.
'''


def set_seed(seed=2020):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
