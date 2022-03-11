"""
The function definition files.
"""
import cv2 as cv
import numpy as np
from scipy.spatial import Voronoi as scipyVoronoi
from scipy.spatial import distance as scipy_distance
from scipy.stats import entropy

def draw_rafts_rh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the right-handed coordinate system
    x pointing right
    y pointing up
    :param numpy array img_bgr: input bgr image in numpy array
    :param numpy array rafts_loc: locations of the rafts
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    x_axis_start = (0, height - 10)
    x_axis_end = (width, height - 10)
    y_axis_start = (10, 0)
    y_axis_end = (10, height)
#    output_img = cv.line(output_img, x_axis_start, x_axis_end, (0, 0, 0), 4)
#    output_img = cv.line(output_img, y_axis_start, y_axis_end, (0, 0, 0), 4)

    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1]),
                               rafts_radii[raft_id], circle_color, circle_thickness)

    return output_img


def draw_rafts_lh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the left-handed coordinate system of openCV
    positive x is pointing right
    positive y is pointing down
    :param numpy array img_bgr: input bgr image in numpy array
    :param numpy array rafts_loc: locations of the rafts
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1]), rafts_radii[raft_id],
                               circle_color, circle_thickness)

    return output_img


def draw_b_field_in_rh_coord(img_bgr, b_orient):
    """
    draw the direction of B-field in right-handed xy coordinate
    :param numpy array img_bgr: bgr image file
    :param float b_orient: orientation of the magnetic B-field, in deg
    :return: bgr image file
    """

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    line_length = 200
    line_start = (width // 2, height // 2)
    line_end = (int(width // 2 + np.cos(b_orient * np.pi / 180) * line_length),
                height - int(height // 2 + np.sin(b_orient * np.pi / 180) * line_length))
    output_img = cv.line(output_img, line_start, line_end, (0, 0, 0), 1)
    return output_img


def draw_raft_orientations_rh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft,
    as indicated by rafts_ori, in a right-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
        # note that the sign in front of the sine term is "+"
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    height - int(rafts_loc[raft_id, 1] +
                                 np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_raft_orientations_lh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft,
    as indicated by rafts_ori, in left-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
        # note that the sign in front of the sine term is "-"
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    int(rafts_loc[raft_id, 1] - np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]))
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_cap_peaks_rh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in right-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param int raft_sym: the symmetry of raft
    :param int cap_offset: the angle between the dipole direction
    and the first capillary peak, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym
    #    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            # note that the sign in front of the sine term is "+"
            line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
            line_end = (int(rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]),
                        height - int(rafts_loc[raft_id, 1] + np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                                    * np.pi / 180) * rafts_radii[raft_id]))
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_cap_peaks_lh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in left-handed coordinate system
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param numpy array rafts_ori: the orientation of rafts, in deg
    :param int raft_sym: the symmetry of raft
    :param int cap_offset: the angle between the dipole direction
    and the first capillary peak, in deg
    :param numpy array rafts_radii: radii of the rafts
    :param int num_of_rafts: num of rafts
    :return bgr image file
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
            line_end = (int(rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]),
                        int(rafts_loc[raft_id, 1] - np.sin((rafts_ori[raft_id] + cap_offset + capID * cap_gap)
                                                           * np.pi / 180) * rafts_radii[raft_id]))
            # note that the sign in front of the sine term is "-"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_raft_num_rh_coord(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    in the right-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1), (rafts_loc[raft_id, 0] - text_size[0] // 2,
                                                               height - (rafts_loc[raft_id, 1] + text_size[1] // 2)),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_raft_number_lh_coord(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    in the left-handed coordinate
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :param int num_of_rafts: num of rafts
    :return: bgr image file
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1] // 2),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_frame_info(img_bgr, time_step_num, distance, orientation, b_field_direction, rel_orient):
    """
    draw information on the output frames
    :param numpy array img_bgr: input bgr image
    :param int time_step_num: current step number
    :param float distance: separation distance between two rafts
    :param float orientation: orientation of the raft 0 (same for all rafts)
    :param float b_field_direction: orientation of the B-field
    :param float rel_orient: relative orientation phi_ji
    :return: bgr image
    """
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 1
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    text_size, _ = cv.getTextSize(str(time_step_num), font_face, font_scale, font_thickness)
    line_padding = 2
    left_padding = 20
    top_padding = 20
    output_img = cv.putText(output_img, 'time step: {}'.format(time_step_num), (left_padding, top_padding), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'distance: {:03.2f}'.format(distance),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 1), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'orientation of raft 0: {:03.2f}'.format(orientation),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 2), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'B_field_direction: {:03.2f}'.format(b_field_direction),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 3), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'relative orientation phi_ji: {:03.2f}'.format(rel_orient),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 4), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('magnetic_dipole_force: {}'.format(magnetic_dipole_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 5), font_face,
    #                         font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('capillary_force: {}'.format(capillary_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 6),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('hydrodynamic_force: {}'.format(hydrodynamic_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 7),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('B-field_torque: {}'.format(B-field_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 8),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('mag_dipole_torque: {}'.format(mag_dipole_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 9),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('cap_torque: {}'.format(cap_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 10),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)

    return output_img


def draw_frame_info_many(img_bgr, time_step_num, hex_order_avg_norm, hex_order_norm_avg, entropy_by_distances):
    """
    draw information on the output frames
    :param numpy array img_bgr: input bgr image
    :param int time_step_num: current step number
    :param float hex_order_avg_norm: the norm of the average of the hexatic order parameter
    :param float hex_order_norm_avg: the average of the norm of the hexatic order parameter
    :param float entropy_by_distances: entropy by neighbor distances
    :return: bgr image
    """
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 1
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    text_size, _ = cv.getTextSize(str(time_step_num), font_face, font_scale, font_thickness)
    line_padding = 2
    left_padding = 20
    top_padding = 20
    output_img = cv.putText(output_img, 'time step: {}'.format(time_step_num), (left_padding, top_padding), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'hex_order_avg_norm: {:03.2f}'.format(hex_order_avg_norm),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 1), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'hex_order_norm_avg: {:03.2f}'.format(hex_order_norm_avg),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 2), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'entropy_by_distances: {:03.2f}'.format(entropy_by_distances),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 3), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('velocity_torque_coupling: {}'.format(velocity_torque_coupling)),
    #                         (10, 10 + (text_size[1] + line_padding) * 4 ), font_face, font_scale, font_color,
    #                         font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('magnetic_dipole_force: {}'.format(magnetic_dipole_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 5), font_face, font_scale,font_color,
    #                         font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('capillary_force: {}'.format(capillary_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 6), font_face, font_scale,
    #                         font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('hydrodynamic_force: {}'.format(hydrodynamic_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 7), font_face, font_scale,
    #                         font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('B-field_torque: {}'.format(B-field_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 8), font_face, font_scale,
    #                         font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('mag_dipole_torque: {}'.format(mag_dipole_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 9), font_face, font_scale,
    #                         font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('cap_torque: {}'.format(cap_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 10), font_face, font_scale,
    #                         font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_voronoi_rh_coord(img_bgr, rafts_loc):
    """
    draw Voronoi patterns in the right-handed coordinates
    :param numpy array img_bgr: the image in bgr format
    :param numpy array rafts_loc: the locations of rafts
    :return: bgr image file
    """
    height, width, _ = img_bgr.shape
    points = rafts_loc
    points[:, 1] = height - points[:, 1]
    vor = scipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255, 0, 0)
    for x_pos, y_pos in zip(vor.vertices[:, 0], vor.vertices[:, 1]):
        output_img = cv.circle(output_img, (int(x_pos), int(y_pos)), vertex_size, vertex_color)

    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])),
                                 (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color,
                                 edge_thickness)

    center = points.mean(axis=0)
    for point_idx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = points[point_idx[1]] - points[point_idx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = points[point_idx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 200
            output_img = cv.line(output_img, (int(vor.vertices[i, 0]), int(vor.vertices[i, 1])),
                                 (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img


def fft_general(sampling_rate, signal):
    """
    given sampling rate and signal,
    output frequency vector and one-sided power spectrum
    :param numpy array signal: the input signal in 1D array
    :param float sampling_rate: sampling rate in Hz
    :return: frequencies, one-sided power spectrum, both numpy array
    """
    #    sampling_interval = 1/sampling_rate # unit s
    #    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(signal)  # total number of frames
    fft = np.fft.fft(signal)
    p2 = np.abs(fft / sampling_length)
    p1 = p2[0:int(sampling_length / 2) + 1]
    p1[1:-1] = 2 * p1[1:-1]  # one-sided power spectrum
    frequencies = sampling_rate / sampling_length * np.arange(0, int(sampling_length / 2) + 1)

    return frequencies, p1


def adjust_phases(phases_input):
    """
    adjust the phases to get rid of the jump of 360
    when it crosses from -180 to 180, or the reverse
    adjust single point anomaly.
    :param numpy array phases_input: initial phases, in deg
    :return: ajusted phases, in deg
    """
    phase_diff_threshold = 200

    phases_diff = np.diff(phases_input)

    index_neg = phases_diff < -phase_diff_threshold
    index_pos = phases_diff > phase_diff_threshold

    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)

    phase_diff_corrected = phases_diff.copy()
    phase_diff_corrected[insertion_indices_neg[0]] += 360
    phase_diff_corrected[insertion_indices_pos[0]] -= 360

    phases_corrected = phases_input.copy()
    phases_corrected[1:] = phase_diff_corrected[:]
    phases_adjusted = np.cumsum(phases_corrected)

    return phases_adjusted


def shannon_entropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """

    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero(c_normalized)]  # gives 1D array
    h = -sum(c_normalized_nonzero * np.log2(c_normalized_nonzero))  # unit in bits
    return h


def square_spiral(num_of_rafts, spacing, origin):
    """
    initialize the raft positions using square spirals
    ref:
    https://stackoverflow.com/questions/398299/looping-in-a-spiral
    :param int num_of_rafts: number of rafts
    :param float spacing: the spacing between lines
    :param numpy array origin: numpy array of float
    :return: locations of rafts in square spiral
    """
    raft_locations = np.zeros((num_of_rafts, 2))
#    X =Y = int(np.sqrt(num_of_rafts))
    x = y = 0
    # dx = 0
    # dy = -1
    # for i in range(num_of_rafts):
    #
    #     raft_locations[i, :] = np.array([x, y]) * spacing + origin
    #     if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
    #         dx, dy = -dy, dx
    #     x, y = x+dx, y+dy

    # or more explicitly
    dx, dy = 1, 0
    for i in range(num_of_rafts):
        raft_locations[i, :] = np.array([x, y]) * spacing + origin
        if x > 0 and x == 1 - y:
            dx, dy = 0, 1
        elif x > 0 and x == y:
            dx, dy = -1, 0
        elif x < 0 and x == -y:
            dx, dy = 0, -1
        elif x < 0 and x == y:
            dx, dy = 1, 0
        x, y = x+dx, y+dy
    return raft_locations


def hexagonal_spiral(num_of_rafts, spacing, origin):
    """
    initialize the raft positions using square spirals
    ref:
    https://stackoverflow.com/questions/398299/looping-in-a-spiral
    :param int num_of_rafts: number of rafts
    :param float spacing: the spacing between lines
    :param numpy array origin: numpy array of float
    :return: locations of rafts in square spiral
    """
    raft_locations = np.zeros((num_of_rafts, 2))
    a = b = 0
    da, db = 1, 0
    for i in range(num_of_rafts):
        noise = np.random.uniform(low=-1, high=1, size=2) * 0.1 * spacing
        raft_locations[i, :] = np.array([a + b*np.cos(np.pi/3), b*np.sin(np.pi/3)]) * spacing + noise + origin
        if a > 0 and b == 0:
            da, db = -1, +1
        elif a == 0 and b > 0:
            da, db = -1, 0
        elif a < 0 and b == -a:
            da, db = 0, -1
        elif a < 0 and b == 0:
            da, db = 1, -1
        elif a == 0 and b < 0:
            da, db = 1, 0
        elif a > 0 and b == 1 - a:
            da, db = 0, 1
        a, b = a+da, b+db
    return raft_locations


def neighbor_distances_array(raft_locations):
    """
    :param raft_locations: shape: (num of rafts, 2)
    """
    num_of_rafts, _ = raft_locations.shape
    pairwise_distances = scipy_distance.cdist(raft_locations, raft_locations, 'euclidean')
    # Voronoi calculation
    vor = scipyVoronoi(raft_locations)
    all_vertices = vor.vertices
    neighbor_pairs = vor.ridge_points  # row# is the index of a ridge,
    # columns are the two point# that correspond to the ridge

    ridge_vertex_pairs = np.asarray(vor.ridge_vertices)  # row# is the index of a ridge,
    # columns are two vertex# of the ridge

    neighbor_distances = []
    # calculate hexatic order parameter and entropy by neighbor distances
    for raftID in np.arange(num_of_rafts):
        # raftID = 0
        r_i = raft_locations[raftID, :]  # unit: micron

        # neighbors of this particular raft:
        ridge_indices0 = np.nonzero(neighbor_pairs[:, 0] == raftID)
        ridge_indices1 = np.nonzero(neighbor_pairs[:, 1] == raftID)
        ridge_indices = np.concatenate((ridge_indices0, ridge_indices1), axis=None)
        neighbor_pairs_of_one_raft = neighbor_pairs[ridge_indices, :]
        nns_of_one_raft = np.concatenate((neighbor_pairs_of_one_raft[neighbor_pairs_of_one_raft[:, 0] == raftID, 1],
                                          neighbor_pairs_of_one_raft[neighbor_pairs_of_one_raft[:, 1] == raftID, 0]))
        neighbor_distances.append(pairwise_distances[raftID, nns_of_one_raft])

    return np.concatenate(neighbor_distances)


def neighbor_distances_angles_array(raft_locations):
    """
    :param raft_locations: shape: (num of rafts, 2)
    """
    num_of_rafts, _ = raft_locations.shape
    pairwise_distances = scipy_distance.cdist(raft_locations, raft_locations, 'euclidean')
    # Voronoi calculation
    vor = scipyVoronoi(raft_locations)
    all_vertices = vor.vertices
    neighbor_pairs = vor.ridge_points  # row# is the index of a ridge,
    # columns are the two point# that correspond to the ridge

    # ridge_vertex_pairs = np.asarray(vor.ridge_vertices)  # row# is the index of a ridge, not in use
    # columns are two vertex# of the ridge

    neighbor_distances = []
    neighbor_angles = []
    hex_order_parameters = []
    # calculate hexatic order parameter and entropy by neighbor distances
    for raftID in np.arange(num_of_rafts):
        # raftID = 0
        r_i = raft_locations[raftID, :]  # unit: micron

        # neighbors of this particular raft:
        ridge_indices0 = np.nonzero(neighbor_pairs[:, 0] == raftID)
        ridge_indices1 = np.nonzero(neighbor_pairs[:, 1] == raftID)
        ridge_indices = np.concatenate((ridge_indices0, ridge_indices1), axis=None)
        neighbor_pairs_of_one_raft = neighbor_pairs[ridge_indices, :]
        nns_of_one_raft = np.concatenate((neighbor_pairs_of_one_raft[neighbor_pairs_of_one_raft[:, 0] == raftID, 1],
                                          neighbor_pairs_of_one_raft[neighbor_pairs_of_one_raft[:, 1] == raftID, 0]))
        neighbor_distances.append(pairwise_distances[raftID, nns_of_one_raft])

        # calculate hexatic order parameter of this one raft
        neighbor_locations = raft_locations[nns_of_one_raft, :]
        neighbor_angles_in_rad = np.arctan2(-(neighbor_locations[:, 1] - r_i[1]),
                                            (neighbor_locations[:, 0] - r_i[0]))
        neighbor_angles_in_rad_rezeroed = neighbor_angles_in_rad - neighbor_angles_in_rad.min()
        neighbor_angles.append(np.rad2deg(neighbor_angles_in_rad_rezeroed))
        # negative sign to make angle in the right-handed coordinate

        raft_hexatic_order_parameter = \
            np.cos(neighbor_angles_in_rad * 6).mean() + np.sin(neighbor_angles_in_rad * 6).mean() * 1j
        hex_order_parameters.append(raft_hexatic_order_parameter)

    return np.concatenate(neighbor_distances), np.concatenate(neighbor_angles), np.asarray(hex_order_parameters)


def kl_divergence(p, q):
    """
    calculate the KL divergence of two distributions
    :param p: test distribution
    :param q: target distribution
    """
    p_norm = p/p.sum()
    q_norm = q/q.sum()
    q_norm = q_norm + 1e-9  # 1e-7 is added to avoid division by zero problem
    # 1e-7 also penalize any nonzero probability showing up where it should be zero
    c = p_norm/q_norm
    ind = np.nonzero(p_norm)
    kl_d = np.sum(p_norm[ind] * np.log2(c[ind]))
#    kl_d = entropy(q, qk=p)
    
    return kl_d


# def count_distribution(raft_locations, raft_radius, edges_ndist, edges_x, edges_y):
#     """
#     extracting various counts or distributions from raft locations
#     :param raft_locations: shape: (num of rafts, 2)
#     :param raft_radius:
#     :param edges_ndist:
#     :param edges_x:
#     :param edges_y:
#     :return: counts: dictionary containing count_NDist, count_ODist, count_X, count_Y
#     """
#     neighbor_distances = neighbor_distances_array(raft_locations)
#     count_ndist, _ = np.histogram(neighbor_distances / raft_radius, edges_ndist)
#
#     count_ndist = np.zeros(len(edges_ndist) -1)
#
#     count_x, _ = np.histogram(raft_locations[:, 0] / raft_radius, edges_x)
#     count_y, _ = np.histogram(raft_locations[:, 1] / raft_radius, edges_y)
#
#     counts = {"count_NDist": count_ndist,
#               "count_X": count_x,
#               "count_Y": count_y}
#
#     return counts
#
#
# def divergences_curr_target(curr_dict, target_dict):
#     """
#     Give various KL divergence between the two sets of counts
#     :param: curr_dict: dictionary containing current count_NDist, count_X, count_Y
#     :param: target_dict: dictionary containing target count_NDist, count_X, count_Y
#     """
#     kldiv_ndist = kl_divergence(curr_dict['count_NDist'], target_dict['count_NDist'])
#     kldiv_x = kl_divergence(curr_dict['count_X'], target_dict['count_X'])
#     kldiv_y = kl_divergence(curr_dict['count_Y'], target_dict['count_Y'])
#
#     divergences = {"klDiv_NDist": kldiv_ndist,
#                    "klDiv_X": kldiv_x,
#                    "klDiv_Y": kldiv_y}
#     return divergences
#
#
# def entropies_of_counts(dict_with_counts):
#     """
#     Get the entropies of distributions in the count
#     :param dict_with_counts: dictionary containing count_NDist, count_X, count_Y
#     :return: entropies: dictionary containing entropy_NDist, entropy_X, entropy_Y
#     """
#     entropy_ndist = shannon_entropy(dict_with_counts['count_NDist'])
#     entropy_x = shannon_entropy(dict_with_counts['count_X'])
#     entropy_y = shannon_entropy(dict_with_counts['count_Y'])
#
#     entropies = {"entropy_NDist": entropy_ndist,
#                  "entropy_X": entropy_x,
#                  "entropy_Y": entropy_y}
#
#     return entropies


def count_kldiv_entropy_ndist(raft_locations, raft_radius, edges_ndist, target_dict):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_ndist:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    """
    neighbor_distances = neighbor_distances_array(raft_locations)
    count_ndist, _ = np.histogram(neighbor_distances / raft_radius, edges_ndist)
    kldiv_ndist = kl_divergence(count_ndist, target_dict['count_NDist'])
    entropy_ndist = shannon_entropy(count_ndist)

    dict_ndist = {"count_NDist": count_ndist,
                  "klDiv_NDist": kldiv_ndist,
                  "entropy_NDist": entropy_ndist
                  }
    return dict_ndist

def count_entropy_ndist(raft_locations, raft_radius, edges_ndist):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_ndist:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    """
    neighbor_distances = neighbor_distances_array(raft_locations)
    count_ndist, _ = np.histogram(neighbor_distances / raft_radius, edges_ndist)
#    kldiv_ndist = kl_divergence(count_ndist, target_dict['count_NDist'])
    entropy_ndist = shannon_entropy(count_ndist)

    dict_ndist = {"count_NDist": count_ndist,
#                  "klDiv_NDist": kldiv_ndist,
                  "entropy_NDist": entropy_ndist
                  }
    return dict_ndist


def count_kldiv_entropy_ndist_nangles(raft_locations, raft_radius, edges_ndist, edges_nangles, target_dict):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_ndist:
    :param edges_nangles:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    """
    neighbor_distances, neighbor_angles, hex_ord_paras = neighbor_distances_angles_array(raft_locations)

    count_ndist, _ = np.histogram(neighbor_distances / raft_radius, edges_ndist)
    kldiv_ndist = kl_divergence(count_ndist, target_dict['count_NDist'])
    entropy_ndist = shannon_entropy(count_ndist)

    count_nangles, _ = np.histogram(neighbor_angles, edges_nangles)
    kldiv_nangles = kl_divergence(count_nangles[1:], target_dict['count_NAngles'][1:])   # 0 degree does not count
    entropy_nangles = shannon_entropy(count_nangles)

    dict_ndist_nangles = {"count_NDist": count_ndist,
                          "klDiv_NDist": kldiv_ndist,
                          "entropy_NDist": entropy_ndist,
                          "count_NAngles": count_nangles,
                          "klDiv_NAngles": kldiv_nangles,
                          "entropy_NAngles": entropy_nangles,
                          "hexOrderParas": hex_ord_paras
                          }
    return dict_ndist_nangles


def count_kldiv_entropy_odist(raft_locations, raft_radius, edges_odist, target_dict, arena_center=0):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_odist:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    :param arena_center: 0 - not using arena center as center of mass, 1 - using arena center as center of mass
    """
    if type(arena_center) == int:
        center_of_mass = raft_locations.mean(axis=0, keepdims=True)
    else:
        center_of_mass = arena_center.reshape(1, 2)
    orbiting_distances = scipy_distance.cdist(raft_locations, center_of_mass, 'euclidean')
    count_odist, _ = np.histogram(orbiting_distances / raft_radius, edges_odist)
    kldiv_odist = kl_divergence(count_odist, target_dict['count_ODist'])
    entropy_odist = shannon_entropy(count_odist)

    dict_odist = {"count_ODist": count_odist,
                  "klDiv_ODist": kldiv_odist,
                  "entropy_ODist": entropy_odist
                  }
    return dict_odist


def count_kldiv_entropy_x(raft_locations, raft_radius, edges_x, target_dict):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_x:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    """
    count_x, _ = np.histogram(raft_locations[:, 0] / raft_radius, edges_x)
    kldiv_x = kl_divergence(count_x, target_dict['count_X'])
    entropy_x = shannon_entropy(count_x)

    dict_x = {"count_X": count_x,
              "klDiv_X": kldiv_x,
              "entropy_X": entropy_x
              }
    return dict_x


def count_kldiv_entropy_y(raft_locations, raft_radius, edges_y, target_dict):
    """
    calculate the count/distribution, KL divergence and entropy of neighbor distances
    :param raft_locations:
    :param raft_radius:
    :param edges_y:
    :param target_dict: dictionary containing target count_NDist, count_X, count_Y
    """
    count_y, _ = np.histogram(raft_locations[:, 1] / raft_radius, edges_y)
    kldiv_y = kl_divergence(count_y, target_dict['count_Y'])
    entropy_y = shannon_entropy(count_y)

    dict_y = {"count_Y": count_y,
              "klDiv_Y": kldiv_y,
              "entropy_Y": entropy_y
              }
    return dict_y