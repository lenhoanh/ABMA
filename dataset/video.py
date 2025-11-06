import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .image_reader import load_image
from .util import get_videos, get_videos_kf
from .util import get_clips, get_clips_by_video, get_clips_sf, get_clips_rf, get_clips_sf_and_clips_normal
from .data_types import dict_clip


# PIL Image
# https://github.com/vt-le/astnet
def _process_frame(frame_path, channel, width, height, read_format, transform=None):
    """
    Helper function to process a single frame.
    """
    raw_frame = load_image(
        path=frame_path,
        channel=channel,
        resize_width=width,
        resize_height=height,
        format=read_format
    )
    if transform:
        raw_frame = transform(raw_frame)
    return raw_frame


def _process_clip(frames, channel, width, height, read_format, transform=None, clip_mode='stack', clip_dim=0):
    """
    Helper function to process a single clip.
    """
    processed_frames = [
        _process_frame(frame, channel, width, height, read_format, transform) for frame in frames
    ]
    if clip_mode == 'cat':  # Concatenate frames along the specified dimension
        # Concatenates the given sequence of seq tensors in the given dimension.
        # return an array includes 15 elements (tensors) with size of 256x256
        # clip[i].shape = tensor.size(3,256,256) => cat => clip.shape = tensor.size(15,256,256)
        return torch.cat(processed_frames, dim=clip_dim)
    elif clip_mode == 'stack':  # Stack frames along a new dimension
        # Concatenates a sequence of tensors along a new dimension.
        # return a tensor of num_frames(=10) tensors (transformed images)
        # clip[i].shape = tensor.size(3,256,256) => stack => clip.shape = tensor.size(10,3,256,256)
        return torch.stack(processed_frames, dim=clip_dim)
    else:
        return processed_frames  # Return as a list if no specific mode is defined


class VideoDataset(data.Dataset):
    """
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames liên tiếp.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str):
        """
        param data_clip: dict_clip
        param dataset_path: fullpath to train_set or test_set
        """
        super(VideoDataset, self).__init__()
        self.data_clip = data_clip
        # ============================================
        # self.videos is a list of videos
        # self.videos[i] is a list of paths to frames of the i-th video
        self.videos, self.frame_count = get_videos(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # self.clips is a list of clips (concatenated from multiple videos).
        # self.clips[i]: the i-th clip consists of paths to 5 consecutive frames.
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
        # ============================================
        # self.clips_by_video is a list of clips (grouped by individual videos).
        # self.clips_by_video[i] is a list of clips of the i-th video.
        # self.clips_by_video[i][j] is the j-th clip of the i-th video.
        self.clips_by_video = get_clips_by_video(self.videos, data_clip['num_frames'], data_clip['frame_steps'])
        print('len(self.clips_by_video) = ', len(self.clips_by_video))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])

    def __len__(self):
        if self.data_clip['sample_type'] == 'videos':
            return len(self.videos)
        if self.data_clip['sample_type'] == 'clips':
            return len(self.clips)
        elif self.data_clip['sample_type'] == 'clips_by_video':
            return len(self.clips_by_video)

    def __getitem__(self, index: int):
        sample = None
        if self.data_clip['sample_type'] == 'videos':
            video = self.videos[index]  # a list of frame_paths in index-th video
            sample = self.get_sample_from_video(video)
        elif self.data_clip['sample_type'] == 'clips':
            clip = self.clips[index]  # a list of frame_paths in index-th clip
            sample = self.get_sample_from_clip(clip)
        elif self.data_clip['sample_type'] == 'clips_by_video':
            clips = self.clips_by_video[index]  # a list of clips of index-th video
            sample = self.get_sample_from_clips_of_video(clips)
        return sample

    def get_sample_from_videos_bk(self, index: int):
        """
        Return a video: is a list of frames in a video (TENSOR)
        """
        frames = self.videos[index]  # self.videos[index] is a list of frames in a video
        video = []
        if isinstance(self.data_clip['transform'], transforms.Compose):
            # print("The transform is a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format=self.data_clip['read_format'])
                video.append(self.data_clip['transform'](raw_frame))
        else:
            print("The transform is NOT a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format="CV2")
                video.append(raw_frame)

        return video

    def get_sample_from_video(self, video):
        """
        Return a video: a list of frames in a video (TENSOR)
        """
        # frames = self.videos[index]  # List of frame paths in a video
        transform = self.data_clip['transform']
        is_transform_compose = isinstance(transform, transforms.Compose)
        channel = self.data_clip['channel']
        width, height = self.data_clip['width'], self.data_clip['height']
        read_format = self.data_clip['read_format'] if is_transform_compose else "CV2"

        sample = [
            _process_frame(frame, channel, width, height, read_format, transform if is_transform_compose else None)
            for frame in video
        ]

        return sample

    def get_sample_from_clips_bk(self, index: int):
        """
        Return a clip: (TENSOR)
        - a tensor concatenated by num_frames(=5) frames
        - or a tensor stacked by num_frames(=5) frames
        - or a list of num_frames(=5) frames (tensors)
        """
        frames = self.clips[index]  # self.clips[index] is a clip which includes num_frames(=5) frames
        clip = []
        if isinstance(self.data_clip['transform'], transforms.Compose):
            # print("The transform is a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format=self.data_clip['read_format'])
                clip.append(self.data_clip['transform'](raw_frame))
                del raw_frame  # Release memory after using raw_frame
            if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD_Pred
                # Concatenates the given sequence of seq tensors in the given dimension.
                # return an array includes 15 elements (tensors) with size of 256x256
                # clip[i].shape = tensor.size(3,256,256) => cat => clip.shape = tensor.size(15,256,256)
                clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
                # print('clip.shape =', clip.shape) #  (15,256,256)
            elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                # Concatenates a sequence of tensors along a new dimension.
                # return a tensor of num_frames(=10) tensors (transformed images)
                # clip[i].shape = tensor.size(3,256,256) => stack => clip.shape = tensor.size(10,3,256,256)
                clip = torch.stack(clip, dim=self.data_clip['clip_dim'])
                # print('clip.shape =', clip.shape) # (10,3,256,256)
        else:
            print("The transform is NOT a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format="CV2")
                clip.append(raw_frame)
                del raw_frame  # Release memory after using raw_frame
        return clip

    def get_sample_from_clip(self, clip):
        """
        Return a clip: (TENSOR)
        - A tensor concatenated by num_frames(=5) frames
        - Or a tensor stacked by num_frames(=5) frames
        - Or a list of num_frames(=5) frames (tensors)
        """
        # frames = self.clips[index]  # self.clips[index] is a clip containing num_frames(=5) frames
        transform = self.data_clip['transform']
        is_transform_compose = isinstance(transform, transforms.Compose)
        channel = self.data_clip['channel']
        width, height = self.data_clip['width'], self.data_clip['height']
        read_format = self.data_clip['read_format'] if is_transform_compose else "CV2"

        # Process all frames using _process_frame()
        sample = [
            _process_frame(frame, channel, width, height, read_format, transform if is_transform_compose else None)
            for frame in clip
        ]

        # Combine frames based on clip_mode
        clip_mode = self.data_clip['clip_mode']
        clip_dim = self.data_clip['clip_dim']
        if clip_mode == 'cat':  # Concatenate frames along clip_dim
            sample = torch.cat(sample, dim=clip_dim)
        elif clip_mode == 'stack':  # Stack frames along a new dimension
            sample = torch.stack(sample, dim=clip_dim)

        return sample

    def get_sample_from_clips_by_video_bk(self, index: int):
        """
        Return a video: is a list of clips (TENSOR)
        => may cause "Not enough memory" error in testing
        RuntimeError: [enforce fail at C:\cb\pytorch_1000000000000\work\c10\core\impl\alloc_cpu.cpp:72]
        data. DefaultCPUAllocator: not enough memory: you tried to allocate 3932160 bytes.
        at command line: clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
        """
        video = []
        for i in range(len(self.clips_by_video[index])):  # for each clip in a video index
            clip = []
            frames = self.clips_by_video[index][i]
            if isinstance(self.data_clip['transform'], transforms.Compose):
                # print("The transform is a torchvision.transforms.Compose object.")
                for frame in frames:  # for each frame in a clip
                    raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                           resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                           format=self.data_clip['read_format'])
                    clip.append(self.data_clip['transform'](raw_frame))
                    del raw_frame  # Release memory after using raw_frame
                if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD_Pred
                    clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
                elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                    clip = torch.stack(clip, dim=self.data_clip['clip_dim'])
            else:
                print("The transform is NOT a torchvision.transforms.Compose object.")
                for frame in frames:  # for each frame in a clip
                    raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                           resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                           format="CV2")
                    clip.append(raw_frame)
            video.append(clip)
        return video

    def get_sample_from_clips_of_video(self, clips_of_video):
        """
        Return a video: a list of clips (TENSOR)
        Optimized to handle memory issues during testing.
        """
        sample = []
        clip_mode = self.data_clip['clip_mode']
        clip_dim = self.data_clip['clip_dim']
        transform = self.data_clip['transform']
        is_transform_compose = isinstance(transform, transforms.Compose)
        channel = self.data_clip['channel']
        width, height = self.data_clip['width'], self.data_clip['height']
        read_format = self.data_clip['read_format'] if is_transform_compose else "CV2"

        for frames in clips_of_video:  # Iterate over clips in a video
            # Use helper function to process a single clip
            clip = _process_clip(frames, channel, width, height, read_format,
                                 transform if is_transform_compose else None, clip_mode, clip_dim)
            sample.append(clip)
            # Explicitly release memory for processed clip
            del clip

        return sample


class VideoDataset_sf(VideoDataset):
    """
    VideoDataset_sf: VideoDataset with skip frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip, 
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """
    def __init__(self, data_clip, dataset_path: str, skip_frames: list):
        super().__init__(data_clip, dataset_path)

        # Replace self.clips with clips obtained from get_clips_sf()
        if skip_frames is None:
            skip_frames = [2, 3, 4, 5]
        self.clips = get_clips_sf(self.videos, self.data_clip['num_frames'], skip_frames)
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])


class VideoDataset_sf_and_normal(VideoDataset_sf):
    """
    VideoDataset_sf: VideoDataset with skip frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """
    def __init__(self, data_clip, dataset_path: str, skip_frames: list):
        super().__init__(data_clip, dataset_path, skip_frames)

        # Replace self.clips with clips obtained from get_clips_sf()
        if skip_frames is None:
            skip_frames = [2, 3, 4, 5]
        self.clips_sf, self.clips_normal = get_clips_sf_and_clips_normal(self.videos, self.data_clip['num_frames'],
                                                                         skip_frames)

    def __getitem__(self, index: int):
        if self.data_clip['sample_type'] == 'videos':
            video = self.videos[index]  # a list of frame_paths in index-th video
            sample = self.get_sample_from_video(video)
            return sample
        elif self.data_clip['sample_type'] == 'clips':
            clip_sf = self.clips_sf[index]
            clip_normal = self.clips_normal[index]
            sample_sf, sample_normal = self.get_sample_sf_and_sample_normal(clip_sf, clip_normal)
            return sample_sf, sample_normal
        elif self.data_clip['sample_type'] == 'clips_by_video':
            clips_of_video = self.clips_by_video[index]  # a list of clips of index-th video
            sample = self.get_sample_from_clips_of_video(clips_of_video)
            return sample
        return None

    def get_sample_sf_and_sample_normal(self, clip_sf, clip_normal):
        """
        clip_sf = self.clips_sf[index]
        clip_normal = self.clips_normal[index]

        Return a clip: (TENSOR)
        - a tensor concatenated by num_frames(=5) frames
        - or a tensor stacked by num_frames(=5) frames
        - or a list of num_frames(=5) frames (tensors)
        """
        sample_sf = self.get_sample_from_clip(clip_sf)
        sample_normal = self.get_sample_from_clip(clip_normal)
        return sample_sf, sample_normal

    def __len__(self):
        if self.data_clip['sample_type'] == 'videos':
            return len(self.videos)
        if self.data_clip['sample_type'] == 'clips':
            return len(self.clips_sf)
        elif self.data_clip['sample_type'] == 'clips_by_video':
            return len(self.clips_by_video)


# ===============================================================
class VideoDataset_rf(VideoDataset):
    """
    VideoDataset_rf: VideoDataset with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """
    def __init__(self, data_clip, dataset_path: str, repeat_frames: list):
        super().__init__(data_clip, dataset_path)
        if repeat_frames is None:
            repeat_frames = [2, 3]
        self.clips = get_clips_rf(self.videos, self.data_clip['num_frames'], repeat_frames)
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])


# ===========================================================================
class VideoDataset_kf(VideoDataset):
    """
    VideoDataset_rf: VideoDataset with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """
    def __init__(self, data_clip, dataset_path: str):
        super().__init__(data_clip, dataset_path)

        # Replace the videos with those from get_video_kf()
        # get_videos_kf() remove one of two successive frames in dataset_kf if they exist
        self.videos, _ = get_videos_kf(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # Update the clips using the new videos
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
