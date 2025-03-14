import os
import re
import csv
from pathlib import Path

import requests
from urllib.parse import urlencode

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import scipy.io
import cv2


def load_data_from_file(data_set_name, data_path):
    '''
    Load imported Davis series of scalar 2d fields with optional time counts from .npz file
    '''
    data_set_path = data_path + '/' + data_set_name

    # Load file with data imported from DaVis
    try:
        file_name = data_set_path + '.npz'
        loaded_data = np.load(file_name)
        
        xx = loaded_data['x']
        yy = loaded_data['y']
        data = loaded_data['data']
    except IOError as e:
        print(f'Data file { file_name } not found!')
        raise e
    else:
        print(f'Data file { file_name } loaded with { data.shape[0] } records.')

    # Try to load time counts from data    
    try:
        tt = loaded_data['t']
    except:
        tt = np.linspace(0, data.shape[0] - 1, data.shape[0])
        print('Time data not found, use data indecies instead...')

    return xx, yy, tt, data

def try_load_results(result_path, data_set_name):
    '''
    Try to load results from previous processing
    '''
    try:
        stored_results = np.load(result_path + '/' + data_set_name + '_results.npz', allow_pickle=True)
        parameters = stored_results['parameters']
        results = stored_results['results']
        return parameters, results

    except FileNotFoundError:
        return None, None

def save_process_result(result_path, data_set_name, parameters, results):
    '''
    Save parameters and results of processing
    '''
    # Results path if not exist
    Path(result_path).mkdir(parents=True, exist_ok=True)
    file_name = result_path + '/' + data_set_name + '_results.npz'
    np.savez(file_name, parameters=parameters, results=results)
    print(f'Processing results saved to {file_name} file')

def download_from_yandex_disk(file_link, data_path):
    '''
    Download data files from yande disk if not presented in local disk
    '''
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

    # Get download link
    final_url = base_url + urlencode(dict(public_key=file_link))
    response = requests.get(final_url)
    download_url = response.json()['href']

    print(f'Download link for remote data obtained {download_url}')
    print('Start downloading...')

    # Download file
    download_response = requests.get(download_url)
    
    # Try to parse file name (from https://stackoverflow.com/a/51570425)
    s = download_response.headers['content-disposition']
    file_name = re.findall(r'filename\*=([^;]+)', s, flags=re.IGNORECASE)
    if not file_name:
        file_name = re.findall(r'filename=([^;]+)', s, flags=re.IGNORECASE)
    if "utf-8''" in file_name[0].lower():
        file_name = re.sub("utf-8''", '', file_name[0], flags=re.IGNORECASE)
        #fname = urllib.unquote(fname).decode('utf8')
    else:
        file_name = file_name[0]
    # Clean space and double quotes
    file_name.strip().strip('"')

    # Create data path if not exist
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # Save file to data folder
    with open(data_path + '/' + file_name, 'wb') as f:
        f.write(download_response.content)

    print(f'File "{file_name}" succefully downloaded to "{data_path}"')

def get_normalize_scalar_field(field, max_value):

    # Normalize data to [0: max_value]
    min_val = np.min(field.flatten())
    max_val = np.max(field.flatten())

    field = (field - min_val) * max_value / (max_val - min_val)

    return field

def get_crack_ROI(data, x, y, last_rec_num, threshold_ratio, threshold, otsu = True, crack_location='left'):
    
    im = data[last_rec_num, :, :]

    if otsu:
        # Normalize data to [0: 255]
        im = get_normalize_scalar_field(im, 254)
        im = im.astype(np.uint8)
        ret, crack_ROI = cv2.threshold(im, threshold, 255,  cv2.THRESH_OTSU)
    else:
        crack_ROI = np.where(im >= threshold_ratio, 255, 0).astype(np.uint8)

    #crack_ROI = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    # From https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
    _, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(crack_ROI)

    sizes = stats[:, cv2.CC_STAT_AREA]

    max_blob_size_index = np.argmax(sizes[1:]) + 1

    crack_ROI = np.where(im_with_separated_blobs == max_blob_size_index, 255, 0).astype(np.uint8)

    crack_ROI = cv2.morphologyEx(crack_ROI, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    
    #last_x = crack_ROI.shape[0] - 1
    #first_x = last_x

    # Find first x coordinate of crack
    #temp = []

    #for x in range(crack_ROI.shape[1] - 1, 0, -1):
    #    temp.append(np.count_nonzero(np.diff(np.sign(crack_ROI[:, x]))))
    #    if temp[-1] == 2 and temp[-2] > 2:
    #        first_x = x
    #        break

    #crack_ROI[:, first_x+1:] = 0

    #crack_thin = cv2.ximgproc.thinning(crack_ROI.astype(np.uint8))

    #crack_thin = np.nonzero(crack_thin)
    #crack_thin = np.array(list(zip(crack_thin[0], crack_thin[1])))

    crack_thin = []

    crack_ROI_coords = np.nonzero(crack_ROI)

    # Find center line of crack with weightened average 
    if crack_location == 'left' or crack_location == 'right':
        # For horizontal cracks
        crack_ROI_x = np.unique(crack_ROI_coords[1])
        if crack_location == 'right':
            crack_ROI_x = crack_ROI_x[::-1]
        for coord in crack_ROI_x:
            y_array = crack_ROI_coords[0][crack_ROI_coords[1] == coord]
            if np.any(np.diff(y_array) != 1):
                continue
            w_array = im[y_array, coord]
            crack_thin.append([int(np.round(np.average(y_array, weights=w_array))), coord])

    if crack_location == 'top' or crack_location == 'bottom':
        # For vertical cracks
        crack_ROI_y = np.unique(crack_ROI_coords[0])
        if crack_location == 'bottom':
            crack_ROI_y = crack_ROI_y[::-1]
        for coord in crack_ROI_y:
            x_array = crack_ROI_coords[1][crack_ROI_coords[0] == coord]
            if np.any(np.diff(x_array) != 1):
                continue
            w_array = im[coord, x_array]
            crack_thin.append([coord, int(np.round(np.average(x_array, weights=w_array)))])

    crack_thin = np.array(crack_thin)

    crack_points_num = crack_thin.shape[0]

    # Calcaulte real world crack coordinates
    crack_real_coord = []

    for i in range(crack_points_num):
        crack_real_coord.append([
            x[crack_thin[i, 0], crack_thin[i, 1]],
            y[crack_thin[i, 0], crack_thin[i, 1]],
        ])

    # Calculate length of crack for each point
    length = [ 0 ]
    
    for i in range(1, crack_points_num):
        length.append(
            length[-1] + np.sqrt((crack_real_coord[i][0] - crack_real_coord[i-1][0])**2 + (crack_real_coord[i][1] - crack_real_coord[i-1][1])**2)
        )

    #length.reverse()
    length = np.array(length)

    crack_ROI_norm = np.floor(crack_ROI / 255).astype(np.bool)

    # Add minimum increment in crack length
    length = length + length[1]

    return crack_ROI_norm, crack_thin, length

def process_crack(data, crack_ROI, crack_thin, last_rec_num, threshold_ratio, threshold_repeatitions):
    avgs = []
    maxs = []
    repetitions = 0
    crack_detected = False
    field_in_crack = []
    first_crack_index = -1

    for i in range(last_rec_num):
        field = data[i, :, :]

        # Select ROI of crack in field
        crack_field = np.multiply(field, crack_ROI)
        not_crack_field = np.multiply(field, 1 - crack_ROI)

        # Average data near crack point
        temp = []
        for j in range(crack_thin.shape[0]):
            x_min = max(0, crack_thin[j, 0] - 1)
            x_max = min(crack_thin[j, 0] + 2, field.shape[0] - 1)
            y_min = max(0, crack_thin[j, 1] - 1)
            y_max = min(crack_thin[j, 1] + 2, field.shape[1] - 1)
            avg_near_point = np.max(field[x_min:x_max, y_min:y_max])
            temp.append(avg_near_point)

        field_in_crack.append(temp)

        avg_in_field = np.mean(not_crack_field.flatten())
        max_in_crack = np.max(field_in_crack[-1])

        # Try to detect crack
        if not crack_detected and (max_in_crack > threshold_ratio):
            
            # Increment repetitions
            repetitions = repetitions + 1
            
            if repetitions > threshold_repeatitions:
                # Crack detected
                crack_detected = True
                first_crack_index = i
        else:
            # Reset repititions
            repetitions = 0

        avgs.append(avg_in_field)
        maxs.append(max_in_crack)

    return field_in_crack, np.array(avgs), np.array(maxs), first_crack_index

def get_threshold_exceed_times(time_count, field_in_crack, average_strain, threshold, func='min', direction='forward'):
    '''
    Calculate threshold exceed time for every point in crack
    '''
    records_num = len(field_in_crack)
    crack_points_num = len(field_in_crack[0])

    # Find threshold exceeded times for every point of crack
    times_threshold_exceeded = [[] for _ in range(crack_points_num)]

    for i in range(records_num):
        for j in range(crack_points_num):
            if field_in_crack[i][j] > threshold:
                if len(times_threshold_exceeded[j]) > 50:
                    continue
                times_threshold_exceeded[j].append(time_count[i])

    for j in range(crack_points_num):
        times_threshold_exceeded[j] = filter_outliers(np.array(times_threshold_exceeded[j]))

    result_time_threshold_exceeded = np.zeros((crack_points_num, 1))

    # Get average time for threshold exceeding in crack points
    for j in range(crack_points_num):
        try:
            # Take mean, ranked or minimum time exceed for point
            if func == 'mean':             
                result_time_threshold_exceeded[j] = np.sum(times_threshold_exceeded[j]) / len(times_threshold_exceeded[j])
            elif func == 'rank':
                result_time_threshold_exceeded[j] = times_threshold_exceeded[j][int(len(times_threshold_exceeded[j]) * 0.25)]
            elif func == 'min':
                # result_time_threshold_exceeded[j] = np.min(sorted(times_threshold_exceeded[j])[2:])
                result_time_threshold_exceeded[j] = np.min(times_threshold_exceeded[j])
        except:
            result_time_threshold_exceeded[j] = np.nan

    # Filter exceeding time to get it in accsending order only
    if direction == 'forward':
        for j in range(1, crack_points_num):
            if result_time_threshold_exceeded[j] < result_time_threshold_exceeded[j-1]:
                try:
                    result_time_threshold_exceeded[j] = np.min(times_threshold_exceeded[j][np.where(times_threshold_exceeded[j] > result_time_threshold_exceeded[j-1])])
                except ValueError as e:
                    result_time_threshold_exceeded[j] = result_time_threshold_exceeded[j-1]
    elif direction == 'backward':
        for j in range(crack_points_num-1, 0, -1):
            if result_time_threshold_exceeded[j] < result_time_threshold_exceeded[j-1]:
                try:
                    result_time_threshold_exceeded[j-1] = np.min(times_threshold_exceeded[j][np.where(times_threshold_exceeded[j] < result_time_threshold_exceeded[j-1])])
                except ValueError as e:
                    result_time_threshold_exceeded[j-1] = result_time_threshold_exceeded[j]

    return times_threshold_exceeded, result_time_threshold_exceeded

# From https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def filter_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def roi_scalar_field(data, x, y, points, zoom_x=0.5, zoom_y=1.0):
    '''
    Get x, y, z data to plot countour of scalar field with roi in coords of points
    '''
    # Get x and y coords of points
    xx = x[points[:,0], points[:,1]]
    yy = y[points[:,0], points[:,1]]

    # Calculate min max x and y coords
    x_min = np.min(xx) - np.sign(np.min(xx)) * np.min(xx) * zoom_x
    x_max = np.max(xx) + np.sign(np.max(xx)) * np.max(xx) * zoom_x
    y_min = np.min(yy) - np.sign(np.min(yy)) * np.min(yy) * zoom_y
    y_max = np.max(yy) + np.sign(np.max(yy)) * np.max(yy) * zoom_y

    # Calculate zoomed roi coords
    roi_coords = np.where((x > x_min) & (x < x_max) & (y > y_min) & (y < y_max))

    # Get 2D x, y, and z array to plot countour
    xxx = x[roi_coords[0][0]:roi_coords[0][-1], roi_coords[1][0]:roi_coords[1][-1]]
    yyy = y[roi_coords[0][0]:roi_coords[0][-1], roi_coords[1][0]:roi_coords[1][-1]]
    zzz = data[roi_coords[0][0]:roi_coords[0][-1], roi_coords[1][0]:roi_coords[1][-1]]

    return xx, yy, xxx, yyy, zzz 

def draw_scalar_field(scalar_field, xx = None, yy = None):
    '''
    Draw a scalar field (e.i. normal strain field)
    '''
    plt.figure()
    # If x and y coordinates not defined
    if xx is None and yy is None:
        plt.imshow(scalar_field, cmap='rainbow')
    else:
        # Else draw in defined X and Y coordinates
        plt.contourf(xx, yy, scalar_field)
        plt.colorbar()
        plt.grid()

def draw_crack_ROI(crack_ROI, crack_thin, data, last_rec, crack_location):
    '''
    Draw crack ROI with raw strain field and thinned crack
    '''
    if crack_location == 'bottom' or crack_location == 'top':
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    elif crack_location == 'right' or crack_location == 'left':
        _, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

    ax1.imshow(crack_ROI, cmap='rainbow')

    ax2.imshow(data[last_rec, :, :], cmap='rainbow')
    # Add points of thinned crack
    ax2.scatter(crack_thin[:, 1], crack_thin[:, 0], marker='.', linewidths = 1.0, color='black')

def draw_final_legth_crack(data, x, y, last_rec, crack_thin):
    '''
    Draw normal strain field with final length of crack detected
    '''
    plt.figure()

    # Calculate zoomed crack roi and countour it
    xx, yy, xxx, yyy, zzz = roi_scalar_field(data[last_rec], x, y, crack_thin)

    plt.contourf(xxx, yyy, zzz, levels=255, cmap='rainbow')
    plt.xlabel(r'$\it{x}$, mm')
    plt.ylabel(r'$\it{y}$, mm')
    # Remove the minus from the
    xticks = [x for x in plt.xticks()[0] if x > plt.xlim()[0] and x < plt.xlim()[1]]
    yticks = [y for y in plt.yticks()[0] if y > plt.ylim()[0] and y < plt.ylim()[1]]
    plt.xticks(xticks, [f'{abs(x)}' for x in xticks])
    plt.yticks(yticks, [f'{abs(y)}' for y in yticks])
    # Leave four decimal places
    cbar = plt.colorbar(format=StrMethodFormatter('{x:,.4f}'))
    cbar.ax.set_ylabel(r'Maximum normal strain $\it{\epsilon}$')

    # Plot thinned crack points
    plt.plot(xx, yy, 'k-') 

def draw_crack_detected_strain_field(data, x, y, first_crack_index, crack_thin, crack_ROI=None, crack_location=None):
    '''
    Draw strain field for first moment crack is detected
    '''
    plt.figure()

    # Get first moment crack detected field of normal strain
    field = data[first_crack_index]
    # Get mask to highlight field in crack roi
    mask = np.ones(field.shape)
    mask[crack_ROI == False] = 0.1 

    field = np.multiply(field, mask)

    # Calculate zoomed crack roi and countour it
    _, _, xxx, yyy, zzz = roi_scalar_field(field, x, y, crack_thin)

    plt.contourf(xxx, yyy, zzz, levels=255, cmap='rainbow')
    plt.xlabel(r'$\it{x}$, mm')
    plt.ylabel(r'$\it{y}$, mm')
    # Remove the minus from the
    xticks = [x for x in plt.xticks()[0] if x > plt.xlim()[0] and x < plt.xlim()[1]]
    yticks = [y for y in plt.yticks()[0] if y > plt.ylim()[0] and y < plt.ylim()[1]]
    plt.xticks(xticks, [f'{abs(x)}' for x in xticks])
    plt.yticks(yticks, [f'{abs(y)}' for y in yticks])
    # Leave four decimal places
    cbar = plt.colorbar(format=StrMethodFormatter('{x:,.4f}'))
    cbar.ax.set_ylabel(r'Maximum normal strain $\it{\epsilon}$')

def draw_avg_max_normal_strain(time_counts, avg_normal_strain, max_normal_strain):
    '''
    Draw plot average normal strain, maximum normal strain and
    ratio average to maximum normal strain vs time
    '''
    plt.figure()
    plt.plot(time_counts, avg_normal_strain, 'b-', label='Average in whole field')
    plt.plot(time_counts, max_normal_strain, 'r-', label='Maximum in crack ROI')
    plt.xlabel('Time, s')
    plt.ylabel(r'Maximum normal strain $\it{\epsilon}$')
    plt.legend()
    plt.grid()
    
    ax2 = plt.twinx()
    ax2.plot(time_counts, max_normal_strain / avg_normal_strain, 'g-', label='Maxs to avgs ratio')
    plt.ylabel('Maxs to avgs ratio', color='green')
    ax2.tick_params(axis='y', color='green', labelcolor='green')
    plt.grid()

def draw_threshold_exceeded_times(time_counts, crack_length, field_in_crack, times_threshold_exceeded, avg_times_threshold_exceeded):
    '''
    Draw threshold exceeded times detected for every point of crack 
    '''
    max_time = np.max(avg_times_threshold_exceeded) * 1.1
    indices = np.argwhere(time_counts < max_time)
    time_counts_filtered = time_counts[indices]

    x, y = np.meshgrid(crack_length, time_counts_filtered)

    plt.figure()

    ax = plt.contourf(x, y, field_in_crack[:time_counts_filtered.shape[0]], levels=20, cmap='rainbow')
    cb = plt.colorbar()
    cb.set_label(r'Maximum normal strain $\it{\epsilon}$')

    crack_points_num = len(field_in_crack[0])

    for i in range(crack_points_num):
        times_to_plot = [time for time in times_threshold_exceeded[i] if time < max_time]
        plt.scatter(crack_length[[i]*len(times_to_plot)], times_to_plot)
    plt.plot(crack_length, avg_times_threshold_exceeded, 'k-')
    
    plt.xlabel('Distance from crack end, mm')
    plt.ylabel('Time, s')
    plt.grid()

def draw_crack_length_vs_time(result_time_threshold_exceeded, crack_length, ae_data=None):
    '''
    Draw crack length vs time with AE data if specified
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(result_time_threshold_exceeded, crack_length, 'ko-', label='DIC crack length')
    # ax.plot(result_time_threshold_exceeded[0], crack_length[0], 'ro', label='DIC crack detected time')
    ax.set_xlim([0, None])
    if ae_data is None:
        ax.set_ylim([0, None])
    ax.set_ylabel(r'Crack length $\it{s}_{\rm{DIC}}$, mm') 
    ax.set_xlabel(r'Time $\it{T}$, s')
    ax.grid()

    lines, labels = ax.get_legend_handles_labels()

    # Draw AE data to compare
    if ae_data is not None:
        ax2 = ax.twinx()
        ax2.plot(ae_data[0], ae_data[1], label=r'AE events $\it{N}$')
        ax2.plot(ae_data[2], ae_data[3], 'bo', label='AE crack detected time')
        #ax2.set_yscale('log')
        ax2.set_ylim([0, None])
        ax2.set_ylabel('Events')
        ax2.grid()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

def draw_thresholds_times(dic_data, ae_data, data_set_index):
    '''
    Draw threshold time for AE and DIC
    '''
    time_counts = dic_data[0]
    result_time_threshold_exceeded = dic_data[1]
    avg_normal_strain = dic_data[2]
    max_normal_strain = dic_data[3]
    threshold_ratio = dic_data[4]

    # Setup plots and prefiltering in dependence of data set index
    scale_factor = 2
    sparsity_factor = 1
    filer_outliers = False
    sliding_avg_aperture = 1
    ae_events_log_scale = False

    if data_set_index == 0:
        scale_factor = 3
        sliding_avg_aperture = 5
    elif data_set_index == 1:
        scale_factor = 3
    elif data_set_index == 2:
        scale_factor = 3
        sliding_avg_aperture = 5
    elif data_set_index == 3:
        sparsity_factor = 15
        filer_outliers = True
        sliding_avg_aperture = 30
        ae_events_log_scale = True

    # Calculate zoomed time (threshold time for AE and DIC) difference, max and min
    delta_time = abs(result_time_threshold_exceeded[0] - ae_data[2]) / 2
    min_time = max(min(result_time_threshold_exceeded[0], ae_data[2]) - delta_time * scale_factor, 0)
    max_time = max(result_time_threshold_exceeded[0], ae_data[2]) + delta_time * scale_factor
    
    # Determine DIC data in zoomed data
    dic_time = time_counts[(time_counts > min_time) & (time_counts < max_time)]
    dic_ratio = (max_normal_strain)[np.where((time_counts > min_time) & (time_counts < max_time))[0]]

    # Determine AE data in zoomed data
    if ae_data is not None:
        ae_time = ae_data[0][(ae_data[0] >= min_time) & (ae_data[0] <= max_time)]
        ae_events = ae_data[1][(ae_data[0] >= min_time) & (ae_data[0] <= max_time)]
        ae_moment = ae_data[2]
        ae_threshold = ae_data[3]

    # Filter otliers from max normal strain
    if filer_outliers:
        w = 6
        for i in range(0, len(dic_ratio) - w):
            if not abs(dic_ratio[i] - np.mean(dic_ratio[i:i+w])) < 2 * np.std(dic_ratio[i:i+w]):
                dic_ratio[i] = np.median(dic_ratio[i:i+w])
        #filter_outliers(dic_ratio)

    # Draw crack length vs time figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(dic_time[::sparsity_factor], dic_ratio[::sparsity_factor], 'ko')
        
    # Sliding average on ratio of avg to max normal strain
    if sliding_avg_aperture > 1:
        temp = dic_ratio.copy()
        for i in range(sliding_avg_aperture//2, len(dic_ratio) - sliding_avg_aperture//2):
            temp[i] = np.sum(dic_ratio[i - sliding_avg_aperture//2:i + sliding_avg_aperture//2]) / (sliding_avg_aperture//2 * 2)
        dic_ratio = temp

    if data_set_index == 1:
        # Interpolate dic data
        dic_time_interp = np.linspace(np.min(dic_time), np.max(dic_time), 3)
        dic_rations_interp = np.interp(dic_time_interp, dic_time, dic_ratio)

        # Replace threshold time interpolated strain value with threshold value 
        dic_rations_interp[abs(dic_time_interp - result_time_threshold_exceeded[0]) < 1] = threshold_ratio

        ax.plot(dic_time_interp, dic_rations_interp, 'r-')
    else:
        ax.plot(dic_time[:-sliding_avg_aperture//2], dic_ratio[:-sliding_avg_aperture//2], 'r-')

    ax.axhline(y=threshold_ratio, xmin=0, xmax=result_time_threshold_exceeded[0]/max_time[0], color='r', linestyle='--')
    ax.stem(result_time_threshold_exceeded[0], threshold_ratio, linefmt='r--', markerfmt='ro')
    ax.set_xlim([min_time, max_time])
    ax.set_ylim([0, None])
    ax.set_ylabel(r'Maximum normal strain $\it{\epsilon}$', color='red')
    ax.set_xlabel(r'Time $\it{T}$, s')
    ax.tick_params(axis='y', color='red', labelcolor='red')
    ax.grid()

    # Draw AE data to compare
    if ae_data !=  None:
        ax2 = ax.twinx()
        ax2.plot(ae_time, ae_events, 'b-')
        ax2.axhline(y=ae_data[3], xmin=ae_moment/max_time[0], xmax=1, color='b', linestyle='--')
        ax2.stem(ae_moment, ae_threshold, linefmt='b--', markerfmt='bo')
        ax2.set_xlim([min_time, max_time])
        if ae_events_log_scale:
            ax2.set_yscale('log')
        else:
            ax2.set_ylim([0, None])
        ax2.set_ylabel(r'AE events $\it{N}$', color='blue')
        ax2.tick_params(axis='y', color='blue', labelcolor='blue')
        ax2.grid()
    ticks, _ = plt.xticks()
    ticks = np.array([tick for tick in ticks if tick >= min_time and tick <= max_time])
    idx1 = np.argmin(np.abs(ticks - result_time_threshold_exceeded[0]))
    idx2 = np.argmin(np.abs(ticks - ae_moment))
    ticks[idx1] = result_time_threshold_exceeded[0]
    ticks[idx2] = ae_moment
    # labels = ax.get_xticks().tolist()
    labels = [f'{tick:.0f}' for tick in ticks]
    labels[idx1] = r'$\it{T}_{\rm{DIC}}$'
    labels[idx2] = r'$\it{T}_{\rm{AE}}$'
    plt.xticks(ticks, labels)

def export_to_mat(file_name, mdict):
    scipy.io.savemat(file_name, mdict)

def export_to_csv(file_name, crack_length, result_time_threshold_exceeded, remove_dublicated_times=False):
    if not os.path.exists(EXPORT_PATH):
        os.makedirs(EXPORT_PATH)

    if remove_dublicated_times:
        length_vs_time = np.column_stack((crack_length, result_time_threshold_exceeded))
        length_vs_time = length_vs_time[length_vs_time[:, 1].argsort()]

        unique_times, unique_indicies = np.unique(length_vs_time[:, 1], return_index=True)
        lengths_in_times = np.split(length_vs_time[:, 0], unique_indicies)

        filtered_lengths = [np.max(lengths) for lengths in lengths_in_times[1:]]
        length_vs_time_filtered = np.column_stack((filtered_lengths, unique_times))
        crack_length = length_vs_time_filtered[:, 0]
        result_time_threshold_exceeded = length_vs_time_filtered[:, 1, np.newaxis]

    with open(file_name, 'w', newline='') as csv_file:
        csv_writter = csv.writer(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writter.writerow(['Time, s', 'Crack length, mm'])
        for i in range(len(result_time_threshold_exceeded)):
            csv_writter.writerow([result_time_threshold_exceeded[i, 0], crack_length[i]])
    print(f'Crack length vs time data exported to {file_name}')


if __name__ == '__main__':
    
    DATA_PATH = './data'
    RESULT_PATH = './results'
    EXPORT_PATH = './export'

    DATA_SETS = (
        'R_6_28_12_2021', 
        'O16_12_08_2021', 
        'H_9_2_17_12_2021', 
        'R_Nemo_10_03_2022', 
        'H_6_3_17_12_2021', 
        'H_3_3_26_11_2021',
        'SRUT_#1_2025',
        'SRUT_#2_2025',
        'SRUT_#3_2025')
    REMOTE_DIC_DATA_LINKS = ('https://yadi.sk/d/H1vUyhigblohFA', 'https://yadi.sk/d/rn3MWI0-RfzIcg', 'https://yadi.sk/d/9jzHGOotHuD9xQ', 'https://yadi.sk/d/rMMyPa9YhAxVtw', '', '', '', '', '')
    REMOTE_AE_DATA_LINKS = ('', 'https://yadi.sk/d/4P30t4f32ndk7A', 'https://yadi.sk/d/tipwEmUdy2OTzg', 'https://yadi.sk/d/MfUAJK3Y7eQonA', '', '', '', '', '')

    CRACK_LOCATIONS = ('right', 'bottom', 'left', 'left', 'left', 'left', 'left', 'left', 'left')
    AE_FILES = ('P6.mat', 'Shaft(O16).mat', 'H-9-2.mat', 'Rail(10.03.22).mat', '', '', '', '', '')
    AE_COUNTS = (4, 7, 6, 28, None, None, None, None, None)
    AE_MOMENTS = (6.73, 9.43, 7.58, 1300.3, None, None, None, None, None)
    DIC_MOMENTS = (0, 112.9, 11.59, 1300.3, None, None, None, None, None)
    LAST_RECSS = (7923, 1199, 5758, 2666, 207, 11259, 3090, 3099, 3010)
    THRESHOLD_RATIOS = (0.0025, 0.0015, 0.0017, 0.0025, 0.0017, 0.0017, 0.05, 0.05, 0.05)
    THRESHOLDS = (70, 115, 70, 60, 70, 70, 30, 95, 70)
    USE_OTSUS = (True, False, False, False, False, False, False, False, False)

    # Show plots for calculation results
    SHOW_CRACK_ROI = True
    SHOW_FINAL_CRACK_FIELD = True
    SHOW_CRACK_DETECTED_STRAIN_FIELD = True
    SHOW_AVG_MAX_STRAIN = True
    SHOW_THRESHOLD_EXCEEDED_TIMES = True
    SHOW_CRACK_LENGTH_VS_TIME = True
    SHOW_THRESHOLDS_TIMES = True

    # Export crack length vs time to CSV file
    EXPORT_TO_CSV = True
    REMOVE_DUBLICATED_TIME_IN_EXPORT = True

    # Index of data set to calculate
    DATA_SET_INDEX = 8
    
    # Threshold to find crack
    THRESHOLD_RATIO = THRESHOLD_RATIOS[DATA_SET_INDEX]
    # Threshold exceed repetitions to detect crack
    THRESHOLD_REPETITIONS = 0
    # Direction to filter exceeding time in accsending order
    DIRECTION = 'forward'
    # Force to recalculate results with the same parameters
    FORCE_TO_RECALCULATE = True

    DATA_SET_NAME = DATA_SETS[DATA_SET_INDEX]
    REMOTE_DIC_DATA_LINK = REMOTE_DIC_DATA_LINKS[DATA_SET_INDEX]
    REMOTE_AE_DATA_LINK = REMOTE_AE_DATA_LINKS[DATA_SET_INDEX]

    # Last record to process (before destruction)    
    LAST_REC = LAST_RECSS[DATA_SET_INDEX]
    # Threshold to find crack ROI
    THRESHOLD = THRESHOLDS[DATA_SET_INDEX]
    # Use Otsu to find threshold
    USE_OTSU = USE_OTSUS[DATA_SET_INDEX]
    CRACK_LOCATION = CRACK_LOCATIONS[DATA_SET_INDEX]

    DIC_MOMENT = DIC_MOMENTS[DATA_SET_INDEX]

    # AE data
    AE_FILE = AE_FILES[DATA_SET_INDEX]
    AE_MOMENT = AE_MOMENTS[DATA_SET_INDEX]
    AE_COUNT = AE_COUNTS[DATA_SET_INDEX]

    # Set font size for figures 
    plt.rcParams.update({'font.size': 10})

    print(f'Dataset "{DATA_SET_NAME}" choosed to process..')

    # Try to load previous results
    parameters, results = try_load_results(RESULT_PATH, DATA_SET_NAME)

    # Detect if parameters are changed and recalculation is required
    parameters_changed = True
    if parameters is not None:
        parameters_changed = (float(parameters[0]) != THRESHOLD_RATIO or int(parameters[1]) != THRESHOLD_REPETITIONS or
                              parameters[2] != DIRECTION or int(parameters[3]) != LAST_REC or float(parameters[4]) != THRESHOLD or
                              parameters[5] != str(USE_OTSU) or parameters[6] != CRACK_LOCATION)
    
    # Determine if processing data is required
    process_data = FORCE_TO_RECALCULATE or results == None or parameters_changed

    if results is None:
        print('Previous results no found...')
    if parameters_changed:
        print('Loaded results parameteres are changed, recalculation is required...')
    if FORCE_TO_RECALCULATE:
        print('Processing forced to recalculate...')
    if process_data:
        print('Starting processing...')
    else:
        print('Processing skipped, using loaded previous results...')

    # Load data if required
    if process_data or SHOW_CRACK_ROI or SHOW_CRACK_DETECTED_STRAIN_FIELD or SHOW_FINAL_CRACK_FIELD:
        # Load data from file
        try:
            x_coords, y_coords, time_counts, data = load_data_from_file(DATA_SET_NAME, DATA_PATH)
        except FileNotFoundError:
            # Try to load remote data
            print(f'Local files and raw data from DaVis not found. Try to download data from remote...')
            download_from_yandex_disk(REMOTE_DIC_DATA_LINK, DATA_PATH)
            print('Try to open downloaded file again...')
            try:
                x_coords, y_coords, time_counts, data = load_data_from_file(DATA_SET_NAME, DATA_PATH)
            except Exception as e:
                print(f'Error of loading data set "{DATA_SET_NAME}" - {e}')
                raise e

    # Load AE data from .mat file if filename specified   
    ae_data = None

    if AE_FILE != '':
        try:
            mat = scipy.io.loadmat(DATA_PATH + '/' + AE_FILE)
            ae_data = [mat['time'].flatten(), mat['events'].flatten(), AE_MOMENT, AE_COUNT]
            print(f'AE file "{AE_FILE}" loaded...')
        except FileNotFoundError:
            print(f'AE file "{AE_FILE}" not found, try to download it...')
            download_from_yandex_disk(REMOTE_AE_DATA_LINK, DATA_PATH)
            print(f'Try to load file again...')
            try:
                mat = scipy.io.loadmat(DATA_PATH + '/' + AE_FILE)
                ae_data = [mat['time'].flatten(), mat['events'].flatten(), AE_MOMENT, AE_COUNT]
            except Exception as e:
                print(f'Error of loading AE file "{AE_FILE}" - {e}')

    # Process data
    if process_data:        
        # Cut time for last record
        time_counts = time_counts[:LAST_REC]        

        # Find crack ROI, crack points coordinates and crack length
        crack_ROI, crack_points, crack_length = get_crack_ROI(data, x_coords, y_coords, LAST_REC, THRESHOLD_RATIO, THRESHOLD, USE_OTSU, CRACK_LOCATION)
        
        # For each strain field find field in crack, avg and max field, find first index with crack founded
        field_in_crack, avg_normal_strain, max_normal_strain, first_crack_index = process_crack(data, crack_ROI, crack_points, LAST_REC, THRESHOLD_RATIO, THRESHOLD_REPETITIONS)
        
        # Calculate for each crack point time when threshold is exceeded
        times_threshold_exceeded, result_time_threshold_exceeded = get_threshold_exceed_times(time_counts, field_in_crack, avg_normal_strain, THRESHOLD_RATIO, direction=DIRECTION)

        # Save processing results
        parameters = np.array([THRESHOLD_RATIO, THRESHOLD_REPETITIONS, DIRECTION, LAST_REC, THRESHOLD, USE_OTSU, CRACK_LOCATION])
        results = {'crack_ROI': crack_ROI, 'crack_points': crack_points, 'crack_length': crack_length, 'field_in_crack': field_in_crack,
                   'avg_normal_strain': avg_normal_strain, 'max_normal_strain': max_normal_strain, 'first_crack_index': first_crack_index,
                   'times_threshold_exceeded': times_threshold_exceeded, 'result_time_threshold_exceeded': result_time_threshold_exceeded,
                   'time_counts': time_counts}

        save_process_result(RESULT_PATH, DATA_SET_NAME, parameters, results)
    else:
        # Parsing loaded data from previous results
        crack_ROI = results[()]['crack_ROI']
        crack_points = results[()]['crack_points']
        crack_length = results[()]['crack_length']
        field_in_crack = results[()]['field_in_crack']
        avg_normal_strain = results[()]['avg_normal_strain']
        max_normal_strain = results[()]['max_normal_strain']
        first_crack_index = results[()]['first_crack_index']
        times_threshold_exceeded = results[()]['times_threshold_exceeded']
        result_time_threshold_exceeded = results[()]['result_time_threshold_exceeded']
        time_counts = results[()]['time_counts']

    print(f'Crack founded on {first_crack_index} record - {time_counts[first_crack_index]} s')
    
    # Plotting graphs
    print('Plotting graphs...')

    if SHOW_CRACK_ROI:
        draw_crack_ROI(crack_ROI, crack_points, data, LAST_REC, CRACK_LOCATION)

    if SHOW_FINAL_CRACK_FIELD:
        draw_final_legth_crack(data, x_coords, y_coords, LAST_REC, crack_points)

    if SHOW_CRACK_DETECTED_STRAIN_FIELD:
        draw_crack_detected_strain_field(data, x_coords, y_coords, first_crack_index, crack_points, crack_ROI)

    if SHOW_AVG_MAX_STRAIN:
        draw_avg_max_normal_strain(time_counts, avg_normal_strain, max_normal_strain)

    if SHOW_THRESHOLD_EXCEEDED_TIMES:
        draw_threshold_exceeded_times(time_counts, crack_length, field_in_crack, times_threshold_exceeded, result_time_threshold_exceeded)

    if SHOW_CRACK_LENGTH_VS_TIME:
        draw_crack_length_vs_time(result_time_threshold_exceeded, crack_length, None)

    if SHOW_THRESHOLDS_TIMES:
        dic_data = (time_counts, result_time_threshold_exceeded, avg_normal_strain,
                    max_normal_strain, THRESHOLD_RATIO, DIC_MOMENT)

        if ae_data is not None:
            draw_thresholds_times(dic_data, ae_data, DATA_SET_INDEX)
        else:
            print('Error plotting thresholds times - no AE data loaded!')

    # Export to CSV
    if EXPORT_TO_CSV:
        export_to_csv(
            f'{EXPORT_PATH}/{DATA_SET_NAME}.csv', 
            crack_length, 
            result_time_threshold_exceeded,
            remove_dublicated_times=REMOVE_DUBLICATED_TIME_IN_EXPORT)

    # Export to .mat file crack length data
    #export_to_mat(DATA_SET_NAME + '.mat', {'crack_length': crack_length, 'time': result_time_threshold_exceeded})

    # Show all figures if they were specified
    plt.show()

    print('Script finished')
