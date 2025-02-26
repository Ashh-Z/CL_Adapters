import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_images_from_folders(folders, image_names, labels):
    # Create a figure to hold the images
    fig, axs = plt.subplots(3, 8, figsize=(17, 6))
    # Loop over each row (each folder)
    for row, (folder, label) in enumerate(zip(folders, labels)):
        # Loop over each column (each image)
        for col, image_name in enumerate(image_names):
            image_path = os.path.join(folder, image_name + '.png')
            img = mpimg.imread(image_path)
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
        # Set the row label
        axs[row, 0].set_ylabel(label, fontsize=15, rotation=90, labelpad=60, va='center')

    # Adjust spacing and display the plot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f'/volumes1/vlm-cl/paper/celeb_grad.png', bbox_inches='tight')
    plt.show()



# Example usage:
folders = ['/volumes1/vlm-cl/paper/celeb/Baseline_sc', '/volumes1/vlm-cl/paper/celeb/fp_ex_sc', '/volumes1/vlm-cl/paper/celeb/fp_ix_sc']
image_names = ['99', '94', '86', '60', '40', '21', '12', '7']  # Provide the actual image names without extension
labels = ['Base', 'ExLG', 'IxLG']

plot_images_from_folders(folders, image_names, labels)
