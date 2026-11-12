import numpy as np
import torch

class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points

class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

def random_rotate_z(pc):
    # random roate around z axis
    theta = np.random.uniform(0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    return np.matmul(pc, R)

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def jitter_pc(pc, sigma=0.01):
    """
    Randomly jitter points.
    sigma: std of the gaussian noise
    """
    N, C = pc.shape
    assert C == 3
    clip = 3 * sigma
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    pc = pc + jittered_data
    return pc

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def augment_pc(data):
    data = random_point_dropout(data[None, ...])
    data = random_scale_point_cloud(data)
    data = shift_point_cloud(data)
    data = rotate_perturbation_point_cloud(data)
    data = data.squeeze()
    return data

OMNIOBJECT3D_CLASSES = [
 'anise',
 'antique',
 'apple',
 'asparagus',
 'backpack',
 'ball',
 'bamboo_shoots',
 'banana',
 'battery',
 'beauty_blender',
 'bed',
 'belt',
 'biscuit',
 'book',
 'bottle',
 'bowl',
 'box',
 'boxed_beverage',
 'bread',
 'broad_bean',
 'broccoli',
 'broccolini',
 'brush',
 'brussels_sprout',
 'bucket_noodle',
 'bumbag',
 'bun',
 'burrito',
 'cabinet',
 'cake',
 'calculator',
 'candle',
 'candy',
 'canned_beverage',
 'carrot',
 'cauliflower',
 'chair',
 'cheese',
 'cherry',
 'chess',
 'chicken_leg',
 'chili',
 'china',
 'chinese_chess',
 'chocolate',
 'clock',
 'coconut',
 'conch',
 'corn',
 'cucumber',
 'cup',
 'dice',
 'dinosaur',
 'dish',
 'doll',
 'donut',
 'drawing',
 'drum',
 'dumbbell',
 'dumpling',
 'durian',
 'dustbin',
 'earplug',
 'egg',
 'egg_tart',
 'eraser',
 'facial_cream',
 'fan',
 'fig',
 'fire_extinguisher',
 'flash_light',
 'flower_pot',
 'flute',
 'fork',
 'frisbee',
 'garage_kit',
 'garlic',
 'ginger',
 'glasses',
 'glasses_case',
 'gloves',
 'green_bean_cake',
 'guitar',
 'hair_dryer',
 'hairpin',
 'hamburger',
 'hami_melon',
 'hammer',
 'hand_cream',
 'handbag',
 'hat',
 'haw_thorn',
 'hazelnut',
 'helmet',
 'hot_dog',
 'house',
 'insole',
 'kennel',
 'kettle',
 'keyboard',
 'kite',
 'kiwifruit',
 'knife',
 'laptop',
 'laundry_detergent',
 'lemon',
 'light',
 'lipstick',
 'litchi',
 'longan',
 'loquat',
 'lotus_root',
 'magnet',
 'mango',
 'mangosteen',
 'medicine_bottle',
 'microwaveoven',
 'monitor',
 'mooncake',
 'mouse',
 'mushroom',
 'nipple',
 'onion',
 'orange',
 'ornament',
 'ornaments',
 'oyster',
 'package',
 'pad',
 'pan',
 'pancake',
 'pastry',
 'peach',
 'peanut',
 'pear',
 'pen',
 'picnic_basket',
 'pie',
 'pillow',
 'pineapple',
 'pinecone',
 'pistachio',
 'pitaya',
 'pizza',
 'plant',
 'plug',
 'pomegranate',
 'popcorn',
 'potato',
 'potato_chips',
 'power_strip',
 'projector',
 'puff',
 'pumpkin',
 'razor',
 'red_jujube',
 'red_wine_glass',
 'remote_control',
 'rice_cake',
 'ricecooker',
 'rubik_cube',
 'sandwich',
 'sausage',
 'scissor',
 'shampoo',
 'shaomai',
 'shoe',
 'shrimp',
 'skateboard',
 'soap',
 'sofa',
 'spanner',
 'speaker',
 'squash',
 'starfish',
 'steamed_bun',
 'stool',
 'strawberry',
 'suitcase',
 'sushi',
 'sweet_potato',
 'sword_bean',
 'table',
 'table_tennis_bat',
 'tape_measure',
 'teapot',
 'teddy_bear',
 'thermos',
 'thimble',
 'timer',
 'tissue',
 'tomato',
 'tooth_brush',
 'tooth_paste',
 'toothpick_box',
 'toy_animals',
 'toy_boat',
 'toy_bus',
 'toy_car',
 'toy_motorcycle',
 'toy_plane',
 'toy_plant',
 'toy_train',
 'toy_truck',
 'tvstand',
 'umbrella',
 'vase',
 'waffle',
 'wallet',
 'walnut',
 'watch',
 'water_chestnut',
 'watermelon',
 'whistle',
 'yam',
 'zongzi'
]