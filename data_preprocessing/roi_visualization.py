import numpy as np
import matplotlib.pyplot as plt
from train.train_utils import StructuralSimilarity, PearsonCorrelation
from PIL import Image
import torch
import cv2
from torch import nn
from train.train_utils import norm_image_prediction
import pickle
from sklearn.preprocessing import scale


with open('/netpool/work/gpu-3/users/podguzova/datasets/BOLD5000/bold_roi/CSI3/CSI3_roi_pad.pickle', "rb") as input_file:
    subject = pickle.load(input_file)

with open('/netpool/work/gpu-3/users/podguzova/datasets/BOLD5000/bold_train/bold_CSI3_pad.pickle',
          "rb") as input_file:
  subject = pickle.load(input_file)

all_fmri = []
all_stimuli = []
activation_len = []

for item in subject:
  all_fmri.append(item)

fmri_dataset = np.concatenate(all_fmri, axis=0)
print(np.count_nonzero(fmri_dataset==0))
maximum = np.max(fmri_dataset)
minimum = np.min(fmri_dataset)
print('Maximum value:', maximum)
print('Minimum value:', minimum)
plt.grid(True)
plt.hist(scale(fmri_dataset), bins=100)
plt.ylabel('Number of voxels')
plt.xlabel('Voxel values')
plt.title('CSI2')
plt.show()
# plt.text(0.15, 1300000, 'Mean: {:.3f}'.format(np.mean(fmri_dataset)))
# plt.text(0.15, 1200000, 'Std: {:.3f}'.format(np.std(fmri_dataset)))
# plt.text(0.15, 1100000, 'Min: {:.3f}'.format(np.min(fmri_dataset)))
# plt.text(0.15, 1000000, 'Max: {:.3f}'.format(np.max(fmri_dataset)))
# plt.title('Subject 1')
# plt.text(0.4, 5500000, 'Mean: {:.3f}'.format(np.mean(fmri_dataset)))
# plt.text(0.4, 5000000, 'Std: {:.3f}'.format(np.std(fmri_dataset)))
# plt.text(0.4, 4500000, 'Min: {:.3f}'.format(np.min(fmri_dataset)))
# plt.text(0.4, 4000000, 'Max: {:.3f}'.format(np.max(fmri_dataset)))


mean = np.mean(fmri_dataset)
std = np.std(fmri_dataset)
print('Mean value:', mean)
print('Std value:', std)
fmri_dataset = 2 * (fmri_dataset - minimum) / (maximum - minimum) - 1
plt.hist(fmri_dataset, bins=100)
plt.show()
fmri_dataset = scale(fmri_dataset)
#
# with open('/netpool/work/gpu-3/users/podguzova/datasets/BOLD5000/bold_roi/CSI3_scaled_fmri.pickle', 'wb') as f:
#   pickle.dump(fmri_dataset, f)

plt.hist(fmri_dataset, bins=100)
plt.show()

print('Shape', fmri_dataset .shape)
print('Maximum value:', np.max(fmri_dataset))
print('Minimum value:', np.min(fmri_dataset))


structural_similarity = StructuralSimilarity()


"""---------------------------Test SSIM-------------"""
# helper function to load images
load_images = lambda x: np.asarray(Image.open(x))

# Helper functions to convert to Tensors
tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)

# display imgs
def display_imgs(x, transpose=False, resize=False):
  if resize:
    x=cv2.resize(x, (224, 224))
  if transpose:
    cv2.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
  else:
    plt.imshow(x)
    plt.show()


# The true reference Image
img1 = load_images("/netpool/work/gpu-3/users/podguzova/results/encdec_20210114-141556/images/epoch_0_ground_truth_0.png")[..., :3]

# The False image
img2 = load_images("/netpool/work/gpu-3/users/podguzova/results/encdec_20210114-141556/images/epoch_0_ground_truth_1.png")[..., :3]

# The noised true image
noise = np.random.randint(0, 255, (480, 640, 3))
noisy_img = img1 + noise

print("True Image\n")
display_imgs(img1)

print("\nFalse Image\n")
display_imgs(img2)

print("\nNoised True Image\n")
display_imgs(noisy_img)


ssim = StructuralSimilarity()
pcc = PearsonCorrelation()
cos = nn.CosineSimilarity()

_img1 = tensorify(img1)
_img2 = tensorify(img2)
_img1 = norm_image_prediction(_img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
_img2 = norm_image_prediction(_img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
plt.imshow(_img1.permute(0, 2, 3, 1)[0].numpy())
plt.show()
true_vs_false = ssim(_img1, _img2)
print("True vs False Image SSIM Score:", true_vs_false)
print("True vs False Image PCC Score:", pcc(_img1, _img2))
print("True vs False Image COS Score:", cos(_img1, _img2).mean())

_img1 = tensorify(img1)
_img2 = tensorify(noisy_img)
_img1 = norm_image_prediction(_img1)
_img2 = norm_image_prediction(_img2)
true_vs_false = ssim(_img1, _img2)
print("True vs Noisy Image SSIM Score:", true_vs_false)
print("True vs Noisy Image PCC Score:", pcc(_img1, _img2))
print("True vs Noisy Image COS Score:", cos(_img1, _img2).mean())



# Check SSIM score of True image vs True Image
_img1 = tensorify(img1)
_img1 = norm_image_prediction(_img1)
true_vs_false = ssim(_img1, _img1)
print("True vs True Image SSIM Score:", true_vs_false)
print("True vs True Image PCC Score:", pcc(_img1, _img1))
print("True vs True Image COS Score:", cos(_img1, _img1).mean())
