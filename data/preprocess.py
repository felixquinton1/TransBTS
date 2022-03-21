import pickle
import os
import numpy as np
import nibabel as nib

modalities = ('t1')

# train
train_set = {
        'root': '/home/felix/Bureau/TransBTS/data/Train/',
        'image': 'image',
        'label': 'label',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': '/home/felix/Bureau/TransBTS/data/Valid/',
        'image': 'image',
        'label': 'label',
        'flist': 'valid.txt',
        'has_label': True
        }

test_set = {
        'root': '/home/felix/Bureau/TransBTS/data/Test/',
        'image': 'image',
        'label': 'label',
        'flist': 'test.txt',
        'has_label': True
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(path, name, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'label/lb' + name[2:]), dtype='uint8', order='C')
    images = np.array(nib_load(path + 'image/' + name), dtype='float32', order='C') # [240,240,155]
    images = np.expand_dims(images, 3)
    # images = np.concatenate((images,images,images,images), axis=3)
    output = path + name[:-7] + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(1):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, name) for sub, name in zip(subjects, names)]

    for name in names:

        process_f32b0(root, name, has_label)



if __name__ == '__main__':
    doit(train_set)
    doit(valid_set)
    doit(test_set)

