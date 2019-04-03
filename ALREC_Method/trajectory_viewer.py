"""
This script is used to view all the trajectories, including augmented and abnormal ones.
"""
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os.path

number_of_images = 100
start_frame_number = 400

background_image_path = 'background_image_2.png'

overwrite_background_image = True

# Ref.: https://stackoverflow.com/questions/24731035/python-pil-0-5-opacity-transparency-alpha
opacity_level = 250 # Opaque is 255, input between 0-255

def mouse_move(self,event):
    if event.inaxes and event.inaxes.get_navigate():
        s = event.inaxes.format_coord(event.xdata, event.ydata)
        self.set_message(s)

def make_image_transparent(image):
    """
    Makes the image transparent.
    Re.: https://stackoverflow.com/questions/24731035/python-pil-0-5-opacity-transparency-alpha
    :param image: opened image
    :return: transformed image
    """
    image2 = image.convert('RGBA')
    data_array = image2.getdata()

    newData = []
    for item in data_array:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, opacity_level))
        else:
            newData.append(item)

    image2.putdata(newData)
    return image2


def generate_background_image(input_raw_image_frame_path,
                              frame_name='',
                              frame_starting_number=0,
                              is_caviar_data=False):
    if is_caviar_data:
        image_name_1 = input_raw_image_frame_path + frame_name + str(frame_starting_number) + '.jpg'
    else:
        image_name_1 = input_raw_image_frame_path + str(1).zfill(8) + '.jpg'
    im1 = Image.open(image_name_1)
    im1 = make_image_transparent(im1)

    alpha_value = 1.0 / number_of_images

    for i in range(number_of_images):
        if is_caviar_data:
            image_name_2 = input_raw_image_frame_path + frame_name \
                           + str(i+1+frame_starting_number+start_frame_number) + '.jpg'
        else:
            image_name_2 = input_raw_image_frame_path + str(i+1+start_frame_number).zfill(8) + '.jpg'
        im2 = Image.open(image_name_2)
        im2 = make_image_transparent(im2)
        im1 = Image.blend(im1,im2,alpha_value)
        im1 = make_image_transparent(im1)

    im1.save(background_image_path)


class ImageViewer:
    def __init__(self, input_raw_image_frame_path,
                 frame_name='',
                 frame_starting_number=0,
                 is_caviar_data=False):
        self.fig = plt.figure()
        self.ax = plt.axes()

        plt.rcParams.update({'font.size': 22})

        if overwrite_background_image or not os.path.isfile(background_image_path):
            generate_background_image(input_raw_image_frame_path, frame_name, frame_starting_number, is_caviar_data)

        img_test = plt.imread(background_image_path, format='png')
        self.ax.imshow(ndimage.rotate(img_test, 0))

        def format_coord(x,y):
            return "(x={:.2f}, y={:.2f})".format(x,y)
        self.ax.format_coord=format_coord

        mouse_move_patch = lambda arg: mouse_move(self.fig.canvas.toolbar, arg)
        self.fig.canvas.toolbar._idDrag = self.fig.canvas.mpl_connect('motion_notify_event', mouse_move_patch)

    def add_trajectory(self, x_positions, y_positions, line_width=1, line_color='firebrick'):
        self.ax.plot(x_positions, y_positions, '-', linewidth=line_width, color=line_color)
        self.ax.arrow(x_positions[-2], y_positions[-2],
                      x_positions[-1] - x_positions[-2], y_positions[-1] - y_positions[-2],
                      head_width=5*line_width, head_length=2.5*line_width, fc=line_color, ec=line_color)

    def show_image(self):
        plt.show()

    def save_image(self, image_path_name):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(image_path_name)

# Test
#x = range(100,300)
#trajectory_image.add_trajectory(x,x)
#x = range(300,400)
#trajectory_image.add_trajectory(x,x)
#x = range(50,100)
#trajectory_image.add_trajectory(x,x)
#x = range(20,40)
#trajectory_image.add_trajectory(x,x)

#plt.show()
