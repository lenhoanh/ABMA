import torch
from .memory_module import Memory
import torch.nn as nn

"""
- 1 Encoder
- 2 Memory for past and future, 1 SpatialAttention() cho updated_fea, trước khi vào Decoder
- 1 Decoder predict both past and future frame
 
Input of Decoder: 1024*2

"""

class SpatialAttention(nn.Module):
    """
    Module attention không gian đơn giản. Module này tính toán bản đồ attention từ
    đặc trưng trung bình và đặc trưng cực đại của input. Bản đồ attention sau đó được áp dụng
    để nhân với feature map nhằm làm nổi bật các vùng có thông tin quan trọng.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2

        # Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # tích chập Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Input: x có kích thước (B, C, H, W) = (B, 2048, 32, 32)

        # hai bản đồ có kích thước (B, 1, 32, 32), 
        # đại diện cho độ nổi bật trung bình và cực đại tại mỗi vị trí không gian (h, w).
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, 32, 32)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, 32, 32)

        # Ghép lại và đưa qua conv: (B, 2, 32, 32)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # chuẩn hóa bản đồ attention về khoảng [0, 1]
        # attn_map là bản đồ attention có shape (B, 1, 32, 32)
        attn_map = self.sigmoid(self.conv(x_cat)) # (B, 1, 32, 32)

        # attn_map: được broadcast lên input x(B, 2048, 32, 32) theo chiều kênh
        # Kết quả: mỗi pixel tại (h, w) của toàn bộ 2048 kênh sẽ được điều chỉnh theo mức độ quan trọng tại vị trí đó
        return x * attn_map


class double_conv(nn.Module):
    """
    double_conv (tương tự khối Basic): bao gồm 2 lớp Conv2d với BatchNorm2d và ReLU sau mỗi lớp.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            # trích xuất các đặc trưng cơ bản từ dữ liệu đầu vào
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),

            # đóng vai trò như một bước tinh chỉnh
            # lớp này không làm thay đổi số lượng kênh mà tập trung vào việc làm sâu hơn quá trình học đặc trưng,
            # giúp mạng chi tiết hóa các biểu diễn mà không thay đổi kích thước không gian
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_2(nn.Module):
    """
    double_conv (tương tự khối Basic): bao gồm 2 lớp Conv2d với BatchNorm2d và ReLU sau mỗi lớp.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            # trích xuất các đặc trưng cơ bản từ dữ liệu đầu vào
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),

            # đóng vai trò như một bước tinh chỉnh
            # lớp này không làm thay đổi số lượng kênh mà tập trung vào việc làm sâu hơn quá trình học đặc trưng,
            # giúp mạng chi tiết hóa các biểu diễn mà không thay đổi kích thước không gian
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """
    down(): MODEL 2 gom MaxPool2d() va double_conv() == MODEL 1 gom MaxPool2d() va Basic()
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_2(nn.Module):
    """
    down(): MODEL 2 gom MaxPool2d() va double_conv() == MODEL 1 gom MaxPool2d() va Basic()
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv_2(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        up(): MODEL2 gom ConvTranspose2d()+BatchNorm2d()+ReLU() va double_conv() = MODEL 1 gom ConvTranspose2d()+BatchNorm2d()+ReLU() va Basic()
        """
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.up = nn.Sequential(nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=2,
                                                   padding=1, output_padding=1),
                                nn.BatchNorm2d(in_ch // 2),
                                nn.ReLU(inplace=False)
                                )
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, num_frames=4, num_channels=3):
        super(Encoder, self).__init__()

        # inconv(): la khoi double_conv() (Basic) giong MODEL 1
        self.inc = inconv(num_channels * num_frames, 64)

        # down(): bao gom MaxPool2d va double_conv(Basic)
        # down1(), down2(): giong MODEL 1
        # down3(): khac MODEL 1 (MODEL 1 không có BatchNorm2d() va ReLU() sau lớp Conv2d thứ hai)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down_2(256, 512)

    def forward(self, x):
        # x (bz, 12, 256, 256)
        x1 = self.inc(x)  # (bz, 64, H, W)

        x2 = self.down1(x1)  # (bz, 128, H/2, W/2)
        x3 = self.down2(x2)  # (bz, 256, H/4, W/4)
        x4 = self.down3(x3)  # (bz, 512, H/8, W/8)

        return x4, x1, x2, x3  # Trả về x4 và các skip connections


class Decoder(torch.nn.Module):
    def __init__(self, num_channels=3):
        super(Decoder, self).__init__()

        # inconv(): la khoi double_conv() (Basic) giong MODEL 1
        self.dec = inconv(1024*2, 512)

        # up(): bao gom ConvTranspose2d()+BatchNorm2d()+ReLU() va double_conv()
        # up1(), up2(): giong MODEL 1
        # up3() + outc() + tanh(): giong MODEL 1
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)

        # Hai đầu ra riêng biệt cho quá khứ vs tương lai
        self.outc_past = nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        self.outc_future = nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        
        self.tanh = nn.Tanh()

    def forward(self, x, x1, x2, x3):
        # x (4, 1024*2, 32, 32).
        x = self.dec(x)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Dự đoán hai đầu ra
        predict_past = self.tanh(self.outc_past(x))
        predict_future = self.tanh(self.outc_future(x))

        predict = {"past": predict_past, "future": predict_future}
        return predict


class convAE(torch.nn.Module):
    def __init__(self, num_channels=3, num_frames=4, kernel_size=7):
        super(convAE, self).__init__()

        self.encoder = Encoder(num_frames, num_channels)
        self.decoder = Decoder(num_channels)
        
        # Hai bộ nhớ riêng biệt
        self.memory_past = Memory()
        self.memory_future = Memory()

        # Định nghĩa module attention không gian
        self.attention = SpatialAttention(kernel_size)

    def forward(self, x, keys, train=True, is_pseudo=False):
        """
        x.shape = torch.Size([bz, 12, 256, 256])
        keys_past.shape = torch.Size([10, 512])
        keys_future.shape = torch.Size([10, 512])
        fea.shape: torch.Size([bz, 512, 32, 32])
        """
        keys_past = keys["past"].to(x.device)
        keys_future = keys["future"].to(x.device)

        # fea (bz, 512, 32, 32)
        fea, skip1, skip2, skip3 = self.encoder(x)
        if train:
            # Cập nhật hai bộ nhớ: 
            # updated_fea_past (bz,1024,32,32) = fea + concat_memory_past
            updated_fea_past, keys_past, loss_separate_past, loss_compact_past, _ = self.memory_past(
                fea, keys_past, train, is_pseudo)
            
            # updated_fea_future (bz,1024,32,32) = fea + concat_memory_future
            updated_fea_future, keys_future, loss_separate_future, loss_compact_future, _ = self.memory_future(
                fea, keys_future, train, is_pseudo)

            # Kết hợp đặc trưng từ hai bộ nhớ (bz,1024+1024,32,32)
            updated_fea = torch.cat([updated_fea_past, updated_fea_future], dim=1)

            # Áp dụng attention không gian để tập trung vào các vùng quan trọng
            updated_fea = self.attention(updated_fea)

            # Dự đoán từ decoder
            predict = self.decoder(updated_fea, skip1, skip2, skip3)

            # Tổng hợp predict, keys, loss
            keys = {'past': keys_past, 'future': keys_future}
            loss_separate = {'past': loss_separate_past, 'future': loss_separate_future}
            loss_compact = {'past': loss_compact_past, 'future': loss_compact_future}

            return predict, fea, keys, loss_separate, loss_compact
        else:
            # Trong inference, không tính loss_separate
            # updated_fea_past (bz,1024,32,32) = fea + concat_memory_past
            updated_fea_past, keys_past, loss_compact_past, _ = self.memory_past(
                fea, keys_past, train, is_pseudo)
            
            # updated_fea_future (bz,1024,32,32) = fea + concat_memory_future
            updated_fea_future, keys_future, loss_compact_future, _ = self.memory_future(
                fea, keys_future, train, is_pseudo)

            # Kết hợp đặc trưng từ hai bộ nhớ
            updated_fea = torch.cat([updated_fea_past, updated_fea_future], dim=1)

            # Áp dụng attention không gian để tập trung vào các vùng quan trọng
            updated_fea = self.attention(updated_fea)

            # Dự đoán từ decoder
            predict = self.decoder(updated_fea, skip1, skip2, skip3)

            # Tổng hợp predict, keys, loss
            keys = {'past': keys_past, 'future': keys_future}
            loss_compact = {'past': loss_compact_past, 'future': loss_compact_future}

            return predict, fea, keys, loss_compact
