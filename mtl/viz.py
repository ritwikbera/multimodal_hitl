import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
import mplcursors
import pickle
import cv2
import pandas as pd 
import torch
import os

np.random.seed(0)
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.figure(1)
fig = plt.gcf()
ax = plt.gca()

def visualize_gaze(model_filename, dataset, df, exp_folder, index):
    '''
    Given the address of a RGB frame, visualize the true and predicted gaze points
    '''
    # random data
    # inp = torch.Tensor(np.random.rand(1,2048))
    # pred = model(inp)[:,:2] # first two values are mean_x and mean_y
    
    # actual data
    pred = model(dataset[index]['obs'].unsqueeze(0))

    # in case of model returning multiple outputs (ensure gaze output tensor is the first one)
    pred = pred[0] if type(pred) in [tuple, list] else pred
    
    # parse predictions (first two values are mean_x and mean_y)
    x_pred, y_pred = list(pred.squeeze().cpu().detach().numpy())
    x_true, y_true = df['gaze_x'].iloc[index], df['gaze_y'].iloc[index]

    # load images
    img_rgb = cv2.imread(df['rgb_addr'].iloc[index])
    img_height = img_rgb.shape[0]
    img_width = img_rgb.shape[1]

    # add gaze mark to image
    # green, ground truth
    cv2.circle(
        img_rgb,(int(x_true*img_width), int(y_true*img_height)),
        10, (0,255,0), 1)
    # red, prediction
    cv2.circle(
        img_rgb,(int(x_pred*img_width), int(y_pred*img_height)),
        10, (0,0,255), 1) 

    # save to disk
    cv2.imwrite(os.path.join(exp_folder,f'frame{index}.png'), img_rgb)

def plot_embeddings(outputs_file='outputs.npz'):
    outputs = np.load(outputs_file)['outputs']
    labels = np.load(outputs_file)['labels']
    split = np.load(outputs_file)['split']
    
    palette = sns.color_palette("bright", len(np.unique(labels, return_counts=True)[0]))

    data_embedded = TSNE(n_components=2, n_iter=1000, perplexity=30, verbose=1).fit_transform(outputs)
    np.savez('embeddings.npz', data_embedded=data_embedded, labels=labels)

    sns.scatterplot(data_embedded[:split,0], data_embedded[:split,1], hue=labels[:split], legend='full', palette=palette, alpha=1.0, marker='o')
    sns.scatterplot(data_embedded[split:,0], data_embedded[split:,1], hue=labels[split:], palette=palette, alpha=0.5, marker='D')
    plt.savefig('tsne.png')

def plot_pred_errors(file='pred_errors.pkl'):

    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    errors = data['errors']
    filenames = data['files']

    # use scatterplot so that data index can be fetched by mplcursors
    plt.scatter(np.arange(0, len(errors), 1), np.array(errors))

    return errors, filenames


def onclick(event):
    global data
    ix, iy = event.xdata, event.ydata

    # Calculate, based on the axis extent, a reasonable distance 
    # from the actual point in which the click has to occur (in this case 5%)
    ax = plt.gca()
    dx = 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Check for every point if the click was close enough:
    x = np.arange(0, len(data), 1)
    y = np.array(data)
    for i in range(len(filenames)):
        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
            print(f'Opening filename {filenames[i]}')
            
            # Launch image and do stuff here
            image = mpimg.imread(filenames[i])
            plt.figure(2)
            ax = plt.gca()
            plt.imshow(image)
            ax.grid(False)
            plt.show()

data, filenames = plot_pred_errors()

fig.canvas.mpl_connect('button_press_event', onclick)
mplcursors.cursor(ax, hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(filenames[sel.target.index]))

plt.show()