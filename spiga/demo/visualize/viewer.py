import os
import cv2
import copy
import time
import numpy as np

# Demo libs
import spiga.demo.visualize.plotter as plt


class Viewer:

    def __init__(self, window_title, width=None, height=None, fps=30):
        """
        Initialization of the viewer canvas using width and height in pixels
        :param window_title: The string with the window title to display.
        :param width: The given width in pixels of the window canvas.
        :param height: The given height in pixels of the window canvas.
        :param fps: Frames per second
        """
        # Visualizer parameters
        self.canvas = None
        self.width = width
        self.height = height
        self.window_title = window_title
        self.visualize = False

        # Time variables
        self.fps = fps
        self.fps_inference = 0
        self.fps_mean = 0
        self.fps_lifo = np.zeros(self.fps)
        self.timer = time.time()
        self.frame_cnt = -1

        # Video/Image writer
        self.write = False
        self.video_name = window_title  # Initial name
        self.video_path = None
        self.video_writer = None

        # Plots
        self.plotter = plt.Plotter()
        self.fps_draw_params = {'text_size': 0.75,
                                'text_thick': 2,
                                'coord': (10, 50),
                                'font': cv2.FONT_HERSHEY_SIMPLEX,
                                'color': (255, 255, 255)}

    def start_view(self):
        self._kill_window()
        cv2.namedWindow(self.window_title)
        self.visualize = True

    def record_video(self, video_path, video_name=None):
        self.write = True
        if video_name is not None:
            self.video_name = video_name
        self.video_path = video_path
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        file_name = os.path.join(self.video_path, self.video_name + '.mp4')
        self.video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MP4V'),
                                            self.fps, (int(self.width), int(self.height)))

    def save_canvas(self, file_path=None):
        if file_path is None:
            if self.video_path is None:
                raise ValueError('Path not defined neither video_path is available')
            else:
                file_path = self.video_path

        file_name = os.path.join(file_path, '/%s_%i.jpg' % (self.video_name, self.frame_cnt))
        cv2.imwrite(file_path + file_name, self.canvas)

    def reset_params(self, width, height, window_title, fps=30):
        self.width = width
        self.height = height
        self._kill_window()
        if self.video_name == self.window_title:
            self.video_name = window_title
        self.window_title = window_title
        self.fps = fps

    def close(self):
        if self.write:
            self.video_writer.release()
        self._kill_window()

    def process_image(self, input_img, drawers=(), show_attributes=('fps')):

        # Variables
        image = copy.copy(input_img)
        img_h, img_w, img_ch = image.shape

        # Convert gray scale image to color if needed
        if img_ch == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw features on image
        image = self._draw_over_canvas(image, drawers, show_attributes)

        # Resize image if needed to canvas shape
        if img_w != self.width or img_h != self.height:
            image = cv2.resize(image, (self.width, self.height))

        # Update canvas
        self.canvas = image
        # Visualize FPS
        if 'fps' in show_attributes:
            self._plot_fps()

        # Write the resulting frame
        if self.write:
            self.video_writer.write(self.canvas)

        # Timing loop variables
        loop_time = self._update_timers()
        break_flag = False
        # Visualization
        if self.visualize:
            cv2.imshow(self.window_title, self.canvas)
            sleep_time = int(1000 * (1 / self.fps - loop_time))
            if sleep_time <= 0:
                sleep_time = 1
            if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
                break_flag = True

        self.timer = time.time()
        return break_flag

    def _plot_fps(self):
        # Plot algorithm time
        params = self.fps_draw_params
        cv2.putText(self.canvas, ('FPS: %.2f' % self.fps_mean), params['coord'], params['font'], params['text_size'],
                    params['color'], params['text_thick'], cv2.LINE_AA)

    def _draw_over_canvas(self, image, drawers, show_attributes):
        for drawer in drawers:
            image = drawer.plot_features(image, self.plotter, show_attributes)
        return image

    def _kill_window(self):
        self.visualize = False
        try:
            cv2.destroyWindow(self.window_title)
        except:
            pass

    def _update_timers(self):
        self.frame_cnt += 1
        loop_time = time.time() - self.timer
        self.fps_inference = 1/loop_time
        lifo_idx = self.frame_cnt % self.fps
        self.fps_lifo[lifo_idx] = self.fps_inference
        if lifo_idx == 0:
            self.fps_mean = np.mean(self.fps_lifo)
        return loop_time




