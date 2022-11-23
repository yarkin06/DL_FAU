from cProfile import label
# import os.path
import json
# import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        self.__nsamples = len(self.labels)
        self.__epoch = 0
        self.__batch_num = 0
        if self.__nsamples < self.batch_size or self.batch_size == 0:
            self.batch_size = self.__nsamples

        self.__map = np.arange(self.__nsamples)
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        if self.__batch_num * self.batch_size >= self.__nsamples:
            self.__epoch += 1
            self.__batch_num = 0

        if self.__batch_num == 0 and self.shuffle == True:
            np.random.shuffle(self.__map)

        images = np.zeros((self.batch_size, *tuple(self.image_size)))
        labels = np.zeros(self.batch_size, dtype = int)
        start = self.__batch_num * self.batch_size
        if (self.__batch_num + 1) * self.batch_size <= self.__nsamples:
            for i in range(self.batch_size):
                images[i] = self.augment(np.load(f"{self.file_path}/{self.__map[start+i]}.npy"))
                labels[i] = self.labels[str(self.__map[start+i])]
            self.__batch_num += 1
                    
        elif self.__batch_num * self.batch_size < self.__nsamples:
            last_batch_size = self.__nsamples - self.__batch_num*self.batch_size
            for i in range(last_batch_size):
                images[i] = self.augment(np.load(f"{self.file_path}/{self.__map[start+i]}.npy"))
                labels[i] = self.labels[str(self.__map[start+i])]
            for i in range(self.batch_size - last_batch_size):
                images[last_batch_size+i] = self.augment(np.load(f"{self.file_path}/{self.__map[i]}.npy"))
                labels[last_batch_size+i] = self.labels[str(self.__map[i])]
        # print(start, self.__batch_num, self.batch_size, self.__nsamples, self.__epoch)
            
        #TODO: implement next method
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if img.shape != self.image_size:
            img = np.resize(img, self.image_size)
        if self.mirroring == True:
            if np.random.choice((True, False)):
                img = np.fliplr(img)
            if np.random.choice((True, False)):
                img = np.flipud(img)
        if self.rotation == True:
            n_times = np.random.choice((0, 1, 2, 3))
            img = np.rot90(img, n_times)
        #TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.__epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.labels[str(x)]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        imgs, labs = self.next()
        fig = plt.figure(figsize=(10,10))
        cols = 3
        rows = self.batch_size // 3 + (1 if self.batch_size % 3 else 0)

        for i in range(1, self.batch_size+1):
            img = imgs[i-1]
            lab = self.class_dict[labs[i-1]]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(lab)
        plt.show()
        #TODO: implement show method

