import os
import cv2


def main():
    import argparse
    pars = argparse.ArgumentParser(description='Frames to video converter')
    pars.add_argument('frames', type=str, help='Frames directory')
    pars.add_argument('vidname', type=str, help='Output video name')
    pars.add_argument('-o', '--outpath', type=str, default=None, help='Save record')
    pars.add_argument('--fps', type=int, default=30, help='Frames per second')
    pars.add_argument('--shape', nargs='+', type=int, help='Visualizer shape (W,H)')
    args = pars.parse_args()

    if args.shape:
        if len(args.shape) != 2:
            raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
        else:
            video_shape = tuple(args.shape)
    else:
        video_shape = None

    frames2video(args.frames, args.vidname, video_path=args.outpath, video_shape=video_shape, fps=args.fps)


def frames2video(frames_path, video_name, video_path=None, video_shape=None, fps=30):

    frames_names = sorted(os.listdir(frames_path))

    if video_path is None:
        video_path = frames_path + '/vid_out/'

    if not os.path.exists(video_path):
        os.makedirs(video_path)

    video_file = os.path.join(video_path, video_name + '.mp4')
    if video_shape is None:
        video_writer = None
    else:
        vid_w, vid_h = video_shape
        video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (vid_w, vid_h))

    for frame_name in frames_names:
        if frame_name.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'eps', 'bmp', 'gif']:
            print('File %s format doesnt match with an image ' % frame_name)
            continue

        frame_file = os.path.join(frames_path, frame_name)
        frame = cv2.imread(frame_file)
        if video_writer is None:
            vid_h, vid_w = frame.shape[:2]
            video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (vid_w, vid_h))

        if frame.shape[:2] != (vid_h, vid_w):
            frame = cv2.resize(frame, (vid_w, vid_h))
        video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    main()
