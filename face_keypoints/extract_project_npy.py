import numpy as np
import os

w_list = []
filenames = ['0', '2', '3', '4', '7', '12', '68', '70', '76', '135', '160', '250', '251', '344', '439', '543']

print('Loading projected latents...')
for filename in filenames:
    w_path = f'../face_keypoints/projected/seed{filename}/projected_w.npz'
    if os.path.exists(w_path):
        w = np.load(w_path)['w']
        w_list.append(w)
        print(f'✓ {filename}.png → {w.shape}')
    else:
        print(f'✗ Missing: {filename}.png')

if w_list:
    latents = np.stack(w_list)
    output_path = '../face_keypoints/latent_ffhq.npy'
    np.save(output_path, latents)
    print(f'\\n✅ Saved: {output_path}')
    print(f'   Shape: {latents.shape} (16 images × 18 layers × 512 dims)')
else:
    print('\\n❌ No latents found!')