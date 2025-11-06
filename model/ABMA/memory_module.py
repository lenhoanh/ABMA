import torch
import torch.nn as nn
import numpy as np
import functools
from torch.nn import functional as F


def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu


def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)


def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs - 1):
        result = torch.cat((result, distance(a[i], b)), 0)

    return result


def multiply(x):  # to flatten matrix into a vector
    return functools.reduce(lambda x, y: x * y, x, 1)


def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


def index(batch_size, x):
    idx = torch.arange(0, batch_size).long()
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)


def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t)) / 2 + 1 / 2  # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)

    return torch.sum(sim) / (m * (m - 1))

"""
NT-Xent Loss:
- Ưu điểm: Tận dụng nhiều negative, gradient mượt, hiệu quả trong self-supervised learning.
- Nhược điểm: Cần batch lớn, tính toán phức tạp.
- Áp dụng: Học biểu diễn, bài toán contrastive với nhiều negative.

Soft Margin Triplet Loss:
- Ưu điểm: Gradient liên tục, không cần tinh chỉnh margin.
- Nhược điểm: Khó diễn giải trực quan, có thể chậm cải thiện đối với các mẫu cực kỳ khó.
- Áp dụng: Metric learning, nhận dạng khuôn mặt với dữ liệu phức tạp.

Triplet Loss (Hard Margin):
- Ưu điểm: Rõ ràng, tạo áp lực mạnh khi điều kiện không thoả mãn, dễ hiểu.
- Nhược điểm: Gradient bị "cắt" khi đạt margin, yêu cầu lựa chọn margin và negative mẫu khéo léo.
- Áp dụng: Các bài toán yêu cầu sự phân biệt rõ ràng như face recognition, person re-identification.
"""
def nt_xent_loss_cosine(query, pos, neg, temperature=0.5):
    # Giả sử query, pos, neg có shape (B, N, D)
    B, N, D = query.shape
    # Làm phẳng: (B*N, D)
    q = query.view(B * N, D)
    p = pos.view(B * N, D)
    n = neg.view(B * N, D)

    # Tính cosine similarity
    pos_sim = F.cosine_similarity(q, p)  # (B*N,)
    neg_sim = F.cosine_similarity(q, n)  # (B*N,)

    # Ghép logits: kích thước (B*N, 2)
    logits = torch.stack([pos_sim, neg_sim], dim=1) / temperature
    # Nhãn: giá trị 0 ứng với positive
    labels = torch.zeros(B * N, dtype=torch.long, device=q.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def nt_xent_loss_euclidean(anchor, positive, negative, temperature=0.5):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss) sử dụng khoảng cách Euclidean.
    Hàm này khuyến khích anchor gần với positive hơn là với negative.

    Args:
        anchor: Tensor có shape (batch_size, dim)
        positive: Tensor có shape (batch_size, dim)
        negative: Tensor có shape (batch_size, dim)
        temperature: Hệ số scaling (mặc định = 0.5)

    Returns:
        Loss scalar.
    """
    # Tính khoảng cách Euclidean
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # shape: (batch_size,)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # shape: (batch_size,)

    # Biến khoảng cách thành "độ tương đồng" bằng cách lấy đối dấu
    pos_similarity = -pos_dist
    neg_similarity = -neg_dist

    # Ghép thành logits: mỗi mẫu có 2 giá trị: (positive, negative)
    logits = torch.stack([pos_similarity, neg_similarity], dim=1) / temperature

    # Target: positive có index 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def soft_margin_triplet_loss_euclidean(anchor, positive, negative):
    """
    Soft Margin Triplet Loss với khoảng cách Euclidean.
    Mục tiêu: d(anchor, positive) nhỏ hơn d(anchor, negative).
    Công thức: loss = log(1 + exp(d_pos - d_neg)), với
        d_pos = ||anchor - positive||_2,
        d_neg = ||anchor - negative||_2

    Args:
        anchor: Tensor có shape (batch_size, dim)
        positive: Tensor có shape (batch_size, dim)
        negative: Tensor có shape (batch_size, dim)

    Returns:
        Loss scalar.
    """
    # Tính khoảng cách Euclidean
    d_pos = torch.norm(anchor - positive, p=2, dim=1)
    d_neg = torch.norm(anchor - negative, p=2, dim=1)

    # Cách tính trực tiếp
    loss = torch.log1p(torch.exp(d_pos - d_neg))
    return loss


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, k=1, dim=1)
        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        """
        param mem: Tensor(10,512)
        param max_indices: Tensor(2,1024,1)
        param update_indices: Tensor(2,1,10)
        param score: Tensor(2,1024,10)
        param query: Tensor(2,1024,512)
        """
        m, d = mem.size()
        if train:
            batch_size, _, _ = score.size()
            query_update = torch.zeros((batch_size, m, d))

            for b in range(batch_size):
                for i in range(m):
                    idx = torch.nonzero(max_indices[b].squeeze(1) == i)  # idx: Tensor(21,1)
                    a, _ = idx.size()
                    if a != 0:
                        query_update[b, i] = torch.sum(
                            ((score[b, idx, i] / torch.max(score[b, :, i])) * query[b, idx].squeeze(1)),
                            dim=0)
                    else:
                        query_update[b, i] = 0
        else:
            query_update = torch.zeros((m, d))
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)  # idx: Tensor(21,1)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0
            return query_update
        return query_update

    def get_score(self, mem, query, train):
        if train:
            bs, h, w, d = query.size()  # ([4, 32, 32, 512])
            m, d = mem.size()  # (10,512)

            # bởi vì mem và query đều đã được chuẩn hóa rồi
            # nên phép nhân ma trận tương đương với phép cosine similarity
            score = torch.matmul(query, torch.t(mem))  # b x h x w x m
            score = score.view(bs, h * w, m)  # b x (h x w) x m = (4, 1024=32x32, 10)

            # score_query, là xác suất cho từng vector query đối với tất cả các memory vectors.
            score_query = F.softmax(score, dim=1)  # score_query (4, 1024, 10).

            # score_memory, là xác suất cho mỗi vector memory dựa trên tất cả các query.
            score_memory = F.softmax(score, dim=2)  # score_memory (4, 1024, 10).
        else:
            bs, h, w, d = query.size()  # ([4, 32, 32, 512])
            m, d = mem.size()  # (10,512)

            score = torch.matmul(query, torch.t(mem))  # b x h x w x m
            score = score.view(bs * h * w, m)  # (b x h x w) x m = (4x32x32, 10)

            score_query = F.softmax(score, dim=0)
            score_memory = F.softmax(score, dim=1)
        return score_query, score_memory

    def forward(self, query, keys, train=True, is_pseudo=False):
        """
        param query: size ([4, 512, 32, 32])
        param keys: size (10,512) is memory items
        param train: True
        """
        batch_size, dims, h, w = query.size()  # b X d X h X w ([4, 512, 32, 32])
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d ([4, 32, 32, 512])
        if train:
            # loss_separate: torch.Size([4, 1024]); loss_compact: torch.Size([4, 1024, 512])
            loss_separate, loss_compact, closest_mem_indices = self.gather_loss(query, keys, train)

            # read
            updated_query = self.read(query, keys, train)

            # update
            if not is_pseudo:  # update memory with normal sample only
                updated_memory = self.update(query, keys, train)
            else:
                updated_memory = keys
            return updated_query, updated_memory, loss_separate, loss_compact, closest_mem_indices
        else:
            # loss
            loss_compact, closest_mem_indices = self.gather_loss(query, keys, train)

            # read
            updated_query = self.read(query, keys, train)

            # update
            updated_memory = keys

            return updated_query, updated_memory, loss_compact, closest_mem_indices

    def update(self, query, keys, train):
        """
        param query: size ([2, 32, 32, 512])
        """
        if train:
            batch_size, h, w, dims = query.size()  # b x h x w x d  ([4, 32, 32, 512])
            # softmax_score_query, softmax_score_memory: Tensor(4,32,32,512)
            # train (4, 32x32, 10)
            softmax_score_query, softmax_score_memory = self.get_score(keys, query, train)
            query_reshape = query.contiguous().view(batch_size, h * w, dims)

            # gathering_indices: Tensor(4,1024,1), updating_indices: Tensor(4,1,10)
            _, gathering_indices = torch.topk(softmax_score_memory, k=1, dim=2)
            _, updating_indices = torch.topk(softmax_score_query, k=1, dim=1)

            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            batch_size, m, d = query_update.size()
            query_update = query_update.to(keys.device)

            # chuẩn hóa memory
            for b in range(batch_size):
                keys = F.normalize(query_update[b] + keys, dim=1)
            updated_memory = keys.detach()
        else:
            batch_size, h, w, dims = query.size()  # b X h X w X d  ([4, 32, 32, 512])
            # test (4x32x32, 10)
            softmax_score_query, softmax_score_memory = self.get_score(keys, query, train)
            query_reshape = query.contiguous().view(batch_size * h * w, dims)

            _, gathering_indices = torch.topk(softmax_score_memory, k=1, dim=1)
            _, updating_indices = torch.topk(softmax_score_query, k=1, dim=0)

            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            query_update = query_update.to(keys.device)

            # chuẩn hóa memory
            updated_memory = F.normalize(query_update + keys, dim=1)

        return updated_memory.detach()  # updated_memory: size (10,512)

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
        return pointwise_loss

    def gather_loss(self, query, keys, train):  # keys: Tensors(10, 512)
        batch_size, h, w, dims = query.size()  # b, h, w, d (4, 32, 32, 512)
        if train:
            loss_triplet = torch.nn.TripletMarginLoss(margin=1.0, reduction='none')  # margin=1.0 la khoang cach Euclidean
            loss_mse = torch.nn.MSELoss(reduction='none')

            # Compute softmax scores
            # softmax_score_query, softmax_score_memory: torch.Size([4, 1024, 10])
            softmax_score_query, softmax_score_memory = self.get_score(keys, query, train)

            # Reshape query tensor
            # query_reshape: torch.Size([4, 1024, 512]), gathering_indices: torch.Size([4, 1024, 2])
            query_reshape = query.contiguous().view(batch_size, h * w, dims)

            # Get top-2 indices for memories. Return:
            # Giá trị: Một tensor (4, 1024, 2), chứa 2 giá trị lớn nhất cho mỗi phần tử trong chiều thứ hai (mỗi lớp).
            # Chỉ số: Một tensor (4, 1024, 2), chứa chỉ số của các giá trị lớn nhất trong chiều thứ ba (các lớp).
            _, gathering_indices = torch.topk(softmax_score_memory, k=2, dim=2)

            # Gather positive and negative memories: 1st, 2nd closest memories
            # gathering_indices[:, :, 0]: torch.Size([4,1024]) (closest_mem_indices)
            pos = keys[gathering_indices[:, :, 0]]  # pos:Tensor(4,1024,512)
            neg = keys[gathering_indices[:, :, 1]]  # neg:Tensor(4,1024,512)

            # Compute losses
            top1_loss = loss_mse(query_reshape, pos.detach())  # torch.Size([4, 1024, 512])
            gathering_loss = loss_triplet(query_reshape, pos.detach(), neg.detach())  # torch.Size([2, 1024])
            return gathering_loss, top1_loss, gathering_indices[:, :, 0]  # which are loss_separate, loss_compact
        else:
            loss_mse = torch.nn.MSELoss()

            # Compute softmax scores
            softmax_score_query, softmax_score_memory = self.get_score(keys, query, train)

            # Reshape query tensor
            query_reshape = query.contiguous().view(batch_size * h * w, dims)

            # Get top-1 index for memory
            _, gathering_indices = torch.topk(softmax_score_memory, k=1, dim=1)

            # Gather memory and compute loss
            gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

            return gathering_loss, gathering_indices[:, 0]

    def read(self, query, updated_memory, train):
        """
        param query: size ([4, 32, 32, 512])
        param updated_memory: size (10,512)
        param train: True
        """
        if train:
            # # Get dimensions: batch_size=4, h=32, w=32, dims=512
            batch_size, h, w, dims = query.size()  # b x h x w x d

            # Compute softmax scores: softmax_score_query, softmax_score_memory: Tensor(4,1024,10)
            # 1024 queries, 10 items => softmax_score_query(bz=4, 1024, 10)
            softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query, train)

            # Reshape query tensor: query_reshape: Tensor(4,1024,512)
            query_reshape = query.contiguous().view(batch_size, h * w, dims)

            # Compute concatenated memory
            concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)  # b x (h X w) x d

            # Concatenate query and memory
            updated_query = torch.cat((query_reshape, concat_memory), dim=2)  # b x (h x w) x 2d

            # Reshape and permute updated query tensor
            updated_query = updated_query.view(batch_size, h, w, 2 * dims)
            updated_query = updated_query.permute(0, 3, 1, 2)  # Tensor(4,1024=512*2,32,32)
        else:
            batch_size, h, w, dims = query.size()  # b x h x w x d

            # Compute softmax scores
            softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query, train)

            # Reshape query tensor
            query_reshape = query.contiguous().view(batch_size * h * w, dims)

            # Compute concatenated memory
            concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)  # (b x h x w) x d

            # Concatenate query and memory
            updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b x h x w) x 2d

            # Reshape and permute updated query tensor
            updated_query = updated_query.view(batch_size, h, w, 2 * dims)
            updated_query = updated_query.permute(0, 3, 1, 2)

        return updated_query
