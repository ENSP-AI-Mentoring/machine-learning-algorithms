

# %% [markdown]
# ## Load library

# %%
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from algorithms.vae_model import VAE

from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils

# %%
class ChrimasDataset(Dataset):
    def __init__(self, files_path, transform=None) -> None:
        super(ChrimasDataset, self).__init__()
        self.files_path = np.array(files_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.files_path)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_path = self.files_path[index]
        image = io.imread(image_path)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
# %%
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image/255 # .transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}
# %%
import glob

images_path = glob.glob("../dataset/chrismas/*.jpg")
print(len(images_path))

# %%
print(images_path[0])
# %%
chrismas_dataset = ChrimasDataset(
    images_path, transforms.Compose([
        Rescale(124),
        RandomCrop(100),
        ToTensor()
    ])
)

# %%
len(chrismas_dataset)

# %%
image = next(iter(chrismas_dataset))
print(image["image"].size())

# %%

plt.imshow(image["image"].numpy()*255)
plt.show()

# %%
train_size = int(0.8 * len(chrismas_dataset))
test_size = len(chrismas_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(chrismas_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
len(train_loader), len(test_loader)

# %%
if torch.cuda.is_available():
    print('You use GPU !')
    device = torch.device('cuda')
else:
    print('You use CPU !')
    device = torch.device('cpu')

# %% [markdown]
# ## Model configuration

# %%
in_dim = 28*28 # each image is size 28 * 28
encoder_width = [128, 64]
decoder_width = [64, 128]
latent_dim = 32

# %%
batch_size=256
max_epoch=100
lr= 0.001
weight_decay = 0.0075

# %%
data = next(iter(X_train))
data.size()


# %% [markdown]
# ## Train

# %%
# config model
model = VAE(X_train, X_test, in_dim, encoder_width, decoder_width, latent_dim, device)
# train VAE
hist_loss = model.train(batch_size, max_epoch, lr, weight_decay)

# %% [markdown]
# ## A look at encoder/decoder architecture
# 
# ![](https://miro.medium.com/max/1400/1*Q5dogodt3wzKKktE0v3dMQ@2x.png)

# %%
print(model.encoder)

# %%
print(model.decoder)

# %% [markdown]
# Save loss history

# %%
np.savetxt('./dataset/loss.csv', hist_loss, delimiter=',')

# %% [markdown]
# ## Generate new data

# %% [markdown]
# Let generate an example

# %%
X_generated = model.generate_from_latent_space(1)

# %%
X_generated[0].size() # generated image is a size of input image

# %% [markdown]
# Visualize generated with matplotly
# 
# Seem like 7 -:)

# %%
image = Image.fromarray((X_generated[0]*255).cpu().detach().numpy())
plt.figure(figsize=(3, 3))
plt.imshow(image, cmap='gray')
plt.show()

# %% [markdown]
# Let generate 50 examples

# %%
X_generated = model.generate_from_latent_space(50) # generer 50 images

# %% [markdown]
# Random visualize 16 of these

# %%
image_index = np.random.choice(50, 16)

fig, axis = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(12, 12))
selected_images = X_generated[image_index].cpu().detach().numpy()
for i in range(4):
    for j in range(4):
        image = selected_images[i*4+j]
        axis[i, j].imshow(image, cmap='gray')
plt.show()

# %% [markdown]
# Let generate image from example

# %%
data_input, X_generated = model.generate_from_test_data(2) # generer 2 images

# %%
fig, axis = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12, 12))

image = data_input[0].cpu().detach().numpy()
axis[0, 0].imshow(image, cmap='gray')

image = X_generated[0].cpu().detach().numpy()
axis[0, 1].imshow(image, cmap='gray')

image = data_input[1].cpu().detach().numpy()
axis[1, 0].imshow(image, cmap='gray')

image = X_generated[1].cpu().detach().numpy()
axis[1, 1].imshow(image, cmap='gray')

plt.show()

# %% [markdown]
# Forked Source Code [here](https://github.com/shib0li/VAE-torch)

# %%



