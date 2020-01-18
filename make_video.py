import os
import cv2
from datetime import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', type=str, nargs='+', help='Directories containing video frames')
    parser.add_argument('--dest_dir', type=str, default='outputs/videos', help='Destination directory')
    parser.add_argument('--reverse', action='store_true', help='Flag to add a reverse pass of the frames')
    parser.add_argument('--fps', type=int, default=20, help='Video framerate')
    parser.add_argument('--pref', type=str, default='step', help='File name prefix')
    parser.add_argument('--ext', type=str, default='jpg', help='File extension')
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    vid_path = os.path.join(args.dest_dir, 'video') + datetime.now().strftime('%Y-%m-%d_%H.%M.%S.mp4')

    writer = None
    for directory in args.directories:
        frame = 0
        cont = True
        reversing = False
        while cont:
            fname = os.path.join(directory, '{}{}.{}'.format(args.pref, frame, args.ext))
            if os.path.exists(fname):
                img = cv2.imread(fname)
                if reversing:
                    frame -= 1
                else:
                    frame += 1
            else:
                if args.reverse and not reversing:
                    reversing = True
                    frame -= 1
                else:
                    cont = False
                continue
            if writer is None:
                writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.fps, (img.shape[1], img.shape[0]), True)
            writer.write(img)
    writer.release()


