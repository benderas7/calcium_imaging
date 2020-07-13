"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.initialization import downscale
import caiman
import numpy as np
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR

# Set parameters
DISP_MOVIE = False
COLOR_COMPS = True
SAVE_NAME = 'results_movie.avi'
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    print('Number of components: {}'.format(cnm.estimates.C.shape[0]))
    return cnm


def play_movie_custom(
        estimates, imgs, q_max=99.75, q_min=2, gain_res=1, magnification=1, 
        include_bck=True, frame_range=slice(None, None, None), bpx=0, 
        thr=0., save_movie=False, movie_name='results_movie.avi', 
        display=True, opencv_codec='H264', use_color=False, gain_color=4, 
        gain_bck=0.2):
    """Adapted from caiman/source_extraction/cnmf/estimates.py for 3D video."""
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = caiman.movie(imgs[frame_range])
    else:
        imgs = imgs[frame_range]

    y_rec_color = None
    if use_color:
        cols_c = np.random.rand(estimates.C.shape[0], 1, 3) * gain_color
        cs = np.expand_dims(estimates.C[:, frame_range], -1) * cols_c
        y_rec_color = np.tensordot(estimates.A.toarray(), cs, axes=(1, 0))
        y_rec_color = y_rec_color.reshape(dims + (-1, 3), order='F')
        y_rec_color = np.moveaxis(y_rec_color, -2, 0)

    ac = estimates.A.dot(estimates.C[:, frame_range])
    y_rec = ac.reshape(dims + (-1,), order='F')
    y_rec = np.moveaxis(y_rec, -1, 0)
    if estimates.W is not None:
        ssub_b = int(round(np.sqrt(np.prod(dims) / estimates.W.shape[0])))
        b = imgs.reshape((-1, np.prod(dims)), order='F').T - ac
        if ssub_b == 1:
            b = estimates.b0[:, None] + estimates.W.dot(
                b - estimates.b0[:, None])
        else:
            wb = estimates.W.dot(
                downscale(b.reshape(dims + (b.shape[-1],), order='F'),
                          (ssub_b, ssub_b, 1)).reshape((-1, b.shape[-1]),
                                                       order='F'))
            wb0 = estimates.W.dot(downscale(estimates.b0.reshape(
                dims, order='F'), (ssub_b, ssub_b)).reshape(
                (-1, 1), order='F'))
            b = estimates.b0.flatten('F')[:, None] + (np.repeat(np.repeat(
                (wb - wb0).reshape(((dims[0] - 1) // ssub_b + 1,
                                    (dims[1] - 1) // ssub_b + 1, -1),
                                   order='F'),
                ssub_b, 0), ssub_b, 1)[:dims[0], :dims[1]].reshape(
                (-1, b.shape[-1]), order='F'))
        b = b.reshape(dims + (-1,), order='F')
        b = np.moveaxis(b, -1, 0)
    elif estimates.b is not None and estimates.f is not None:
        b = estimates.b.dot(estimates.f[:, frame_range])
        if 'matrix' in str(type(b)):
            b = b.toarray()
        b = b.reshape(dims + (-1,), order='F')
        b = np.moveaxis(b, -1, 0)
    else:
        b = np.zeros_like(y_rec)
    if bpx > 0:
        b = b[:, bpx:-bpx, bpx:-bpx]
        y_rec = y_rec[:, bpx:-bpx, bpx:-bpx]
        imgs = imgs[:, bpx:-bpx, bpx:-bpx]

    y_res = imgs - y_rec - b
    if use_color:
        if bpx > 0:
            y_rec_color = y_rec_color[:, bpx:-bpx, bpx:-bpx]
        mov = caiman.concatenate(
            (np.repeat(np.expand_dims(imgs - (not include_bck) * b, -1), 3, 3),
             y_rec_color + include_bck * np.expand_dims(b * gain_bck, -1),
             np.repeat(np.expand_dims(y_res * gain_res, -1), 3, 3)), axis=2)
    else:
        mov = caiman.concatenate((imgs[frame_range] - (not include_bck) * b,
                                  y_rec + include_bck * b, y_res * gain_res),
                                 axis=2)
    if not display:
        return mov

    if thr > 0:
        import cv2
        out = None
        if save_movie:
            fourcc = cv2.VideoWriter_fourcc(*opencv_codec)
            out = cv2.VideoWriter(movie_name, fourcc, 30.0,
                                  tuple([int(magnification * s) for s in
                                         mov.shape[1:][::-1]]))
        contours = []
        for a in estimates.A.T.toarray():
            a = a.reshape(dims, order='F')
            if bpx > 0:
                a = a[bpx:-bpx, bpx:-bpx]
            # a = cv2.GaussianBlur(a, (9, 9), .5)
            if magnification != 1:
                a = cv2.resize(a, None, fx=magnification, fy=magnification,
                               interpolation=cv2.INTER_LINEAR)
            ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
            contour, hierarchy = cv2.findContours(
                thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours.append(contour)
            contours.append(
                list([c + np.array([[a.shape[1], 0]]) for c in contour]))
            contours.append(
                list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))

        maxmov = np.nanpercentile(mov[0:10],
                                  q_max) if q_max < 100 else np.nanmax(mov)
        minmov = np.nanpercentile(mov[0:10],
                                  q_min) if q_min > 0 else np.nanmin(mov)
        for iddxx, frame in enumerate(mov):
            if magnification != 1:
                frame = cv2.resize(frame, None, fx=magnification,
                                   fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            frame = np.clip((frame - minmov) * 255. / (maxmov - minmov), 0,
                            255)
            if frame.ndim < 3:
                frame = np.repeat(frame[..., None], 3, 2)
            for contour in contours:
                cv2.drawContours(frame, contour, -1, (0, 255, 255), 1)
            cv2.imshow('frame', frame.astype('uint8'))
            if save_movie:
                out.write(frame.astype('uint8'))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        if save_movie:
            out.release()
        cv2.destroyAllWindows()

    else:
        mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                 save_movie=save_movie, movie_name=movie_name)

    return


def make_movie(cnm, disp_movie, color_comps, save_fn):
    # Get images from load memmap
    images = funcs.load_memmap(cnm.mmap_file)

    # Make video for each ROI
    mov = play_movie_custom(
        cnm.estimates, images, display=disp_movie, use_color=color_comps,
        save_movie=bool(save_fn), movie_name=save_fn)
    return mov


def main(results_dir=COMPILED_DIR, disp_movie=DISP_MOVIE,
         color_comps=COLOR_COMPS, save_name=SAVE_NAME):
    # Load results
    cnm = load_results(results_dir)

    # Make movie
    save_fn = os.path.join(results_dir, save_name)
    make_movie(cnm, disp_movie, color_comps, save_fn)
    return


if __name__ == '__main__':
    main()
