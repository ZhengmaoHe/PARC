import pickle
import os

path = '/home/ubuntu/myProject/PARC/parc_dataset/april272025/iter_3/teaser_2000_2049/teaser_2000_0_opt_dm.pkl'
if os.path.exists(path):
    data = pickle.load(open(path, 'rb'))
    print('Keys:', list(data.keys()))
    print('FPS:', data.get('fps'))
    if 'frames' in data:
        print('Frames shape:', data['frames'].shape)
    else:
        print('No frames')
else:
    print('File not found')
