# draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations
# Ref.: https://matplotlib.org/1.3.1/users/event_handling.html

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas.io.sql as sql
import input_data as data
import trajectory_viewer as tv

class DraggableTrajectory:
    lock = None  # only one can be animated at a time
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.press = None
        self.background = None
        self.rotate_factor = np.pi/180
        self.total_scale_factor = 1
        self.local_scale_factor = 1
        self.scaling_change = 0.2

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.trajectory.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.trajectory.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.trajectory.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidrotate = self.trajectory.figure.canvas.mpl_connect(
            'scroll_event', self.on_rotate)
        self.cidkeypress = self.trajectory.figure.canvas.mpl_connect(
            'key_press_event', self.on_key_press)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.trajectory.axes: return
        if DraggableTrajectory.lock is not None: return
        contains, attrd = self.trajectory.contains(event)
        if not contains: return

        x0, y0 = self.trajectory.get_data()
        self.press = x0, y0, event.xdata, event.ydata
        DraggableTrajectory.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.trajectory.figure.canvas
        axes = self.trajectory.axes
        self.trajectory.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.trajectory.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.trajectory)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_key_press(self, event):
        'on key press will activate the appropriate action depending the key'
        if DraggableTrajectory.lock is not self:
            return
        if event.inaxes != self.trajectory.axes: return

        self.local_scale_factor = 1

        if event.key == '+':
            self.local_scale_factor += self.scaling_change
            self.trajectory_scaling(event)
        elif event.key == '-':
            self.local_scale_factor -= self.scaling_change
            self.trajectory_scaling(event)

        self.total_scale_factor *= self.local_scale_factor

    def trajectory_scaling(self, event):
        'Make the trajectory bigger by the scale factor'
        x0, y0 = self.trajectory.get_data()

        x = np.array(x0).reshape(1,-1)
        y = np.array(y0).reshape(1,-1)

        x = x * self.local_scale_factor
        y = y * self.local_scale_factor

        x = x[0,:]
        y = y[0,:]

        self.trajectory.set_xdata(x)
        self.trajectory.set_ydata(y)

        self.press = x, y, event.xdata, event.ydata

        canvas = self.trajectory.figure.canvas
        axes = self.trajectory.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.trajectory)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableTrajectory.lock is not self:
            return
        if event.inaxes != self.trajectory.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.trajectory.set_xdata(x0+dx)
        self.trajectory.set_ydata(y0+dy)

        canvas = self.trajectory.figure.canvas
        axes = self.trajectory.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.trajectory)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_rotate(self, event):
        'on rotate will rotate the trajectory'
        if DraggableTrajectory.lock is not self:
            return
        if event.inaxes != self.trajectory.axes: return

        if event.button == 'up':
            rot = self.rotate_factor
        elif event.button == 'down':
            rot = -self.rotate_factor
        else:
            return

        x0, y0 = self.trajectory.get_data()

        x = np.array(x0).reshape(1,-1)
        y = np.array(y0).reshape(1,-1)

        #====================================================================================
        # Ref.: https://www.mathworks.com/matlabcentral/answers/
        # 93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point

        # create a matrix of these points, which will be useful in future calculations
        v = np.concatenate((x, y), axis=0)

        # choose a point which will be the center of rotation
        center_index = len(x0) / 2
        x_center = x0[center_index]
        y_center = y0[center_index]

        center = np.array([x_center, y_center]).reshape(-1,1)

        # Rotation matrix
        rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

        # Do rotation
        vo = np.dot(rot_matrix, (v - center)) + center

        # pick out the vectors of rotated x- and y-data
        x0_rot = vo[0,:]
        y0_rot = vo[1,:]
        #====================================================================================

        self.trajectory.set_xdata(x0_rot)
        self.trajectory.set_ydata(y0_rot)

        self.press = x0_rot, y0_rot, event.xdata, event.ydata

        canvas = self.trajectory.figure.canvas
        axes = self.trajectory.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.trajectory)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableTrajectory.lock is not self:
            return

        self.press = None
        DraggableTrajectory.lock = None

        # turn off the rect animation property and reset the background
        self.trajectory.set_animated(False)
        self.background = None

        # redraw the full figure
        self.trajectory.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.trajectory.figure.canvas.mpl_disconnect(self.cidpress)
        self.trajectory.figure.canvas.mpl_disconnect(self.cidrelease)
        self.trajectory.figure.canvas.mpl_disconnect(self.cidmotion)
        self.trajectory.figure.canvas.mpl_disconnect(self.cidrotate)

def extract_and_drag_trajectories(raw_input_file_all,
                                  input_raw_image_frame_path,
                                  raw_input_file_names,
                                  video_data_fps,
                                  generate_graph = False, show_graph = False):
    """
    Extracts the trajectories positions and connects with DraggableTrajectory.
    """
    # Print a starting message
    print("=========================================================")
    print("Generating real abnormal trajectories from raw dataset by")
    print("Translating original normal trajectories.")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    img_test = plt.imread(tv.background_image_path, format='png')
    ax.imshow(img_test)

    trajectories = []

    object_label = 0

    for raw_input_file in raw_input_file_names:
        # Skip if the string is empty
        if not raw_input_file:
            object_label += 1
            continue

        # Convert sqlite dataset to dictionary
        conn = sqlite3.connect(raw_input_file)
        raw_dataset = sql.read_sql('select * from {tn}'.format(tn=data.raw_table_name), conn)
        conn.close()

        # Maximum number of objects
        n_objects = int((raw_dataset.loc[raw_dataset['object_id'].idxmax()])['object_id']) + 1

        for object_id in range(0, n_objects):
            # Extract rows for the particular object_id
            df = raw_dataset.loc[raw_dataset['object_id'] == object_id]

            # Extract trajectory
            x_top_left = df['x_top_left'].values
            y_top_left = df['y_top_left'].values
            x_bottom_right = df['x_bottom_right'].values
            y_bottom_right = df['y_bottom_right'].values

            x = (x_top_left + x_bottom_right) / 2
            y = (y_top_left + y_bottom_right) / 2

            new_handler, = ax.plot(x, y, '-', linewidth=object_label+1, color=['red','limegreen','c'][object_label])
            trajectories.append(new_handler)

        object_label += 1

    dts = []

    for trajectory in trajectories:
        dt = DraggableTrajectory(trajectory)
        dt.connect()
        dts.append(dt)

    def save(event):
        if event.key == 's':
            data.extract_and_put_transformed_data(raw_input_file_all=raw_input_file_all,
                                                  input_raw_image_frame_path=input_raw_image_frame_path,
                                                  raw_input_file_names=raw_input_file_names,
                                                  video_data_fps=video_data_fps,
                                                  changed_trajectories=dts,
                                                  generate_graph=generate_graph, show_graph=show_graph)
            print 'Saved modified trajectories to CSV file.'

    fig.canvas.mpl_connect('key_press_event', save)

    plt.show()

    # Print finishing message
    print("                                               ---> Done!")
    print("=========================================================")
