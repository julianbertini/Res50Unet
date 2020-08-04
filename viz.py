import numpy as np
import matplotlib.pyplot as plt

class Visualize():
    
    def __init__(self):
        pass

    def multi_slice_viewer(self, image_label_pair):
        """ 
            Params:
                image_label_pair - array, containing [image, label] 
        """
        self.remove_keymap_conflicts({'j', 'k'})
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

        for i in range(len(axs)):
            axs[i].volume = image_label_pair[i] 
            axs[i].index = image_label_pair[i].shape[0] // 2
            axs[i].imshow(image_label_pair[i][axs[i].index, :, :])

        fig.canvas.mpl_connect('key_press_event', self.process_key)

    def process_key(self, event):
        fig = event.canvas.figure
        if event.key == 'j':
            self.previous_slice(fig.axes)
        elif event.key == 'k':
            self.next_slice(fig.axes)
        fig.canvas.draw()

    def previous_slice(self, axs):
        for i in range(len(axs)):
            volume = axs[i].volume
            axs[i].index = (axs[i].index - 1) % volume.shape[0]  # wrap around using %
            axs[i].images[0].set_array(volume[axs[i].index, :, :])

    def next_slice(self, axs):
        for i in range(len(axs)):
            volume = axs[i].volume
            axs[i].index = (axs[i].index + 1) % volume.shape[0]
            axs[i].images[0].set_array(volume[axs[i].index, :, :])
            

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

