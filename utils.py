import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from scipy import stats
import matplotlib.cm as cm
import seaborn as sns
from typing import List
import pingouin as pg
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


from natsort import index_natsorted

def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Remove error trials from subject data.
    
    Error trials are trials where the subject made an incorrect key press or 
    failed to complete the sequence correctly. This function filters out these 
    trials to analyze only successful sequence executions.
    
    Parameters
    ----------
    subj : pd.DataFrame
        Subject data with an 'isError' column (0 for correct, 1 for error).
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only trials where isError == 0.
    
    Examples
    --------
    >>> subjs_correct = remove_error_trials(subjs)
    """
    return subj[(subj['isError'] == 0)]

# ============================================================================
# FORCE DISTANCE CALCULATIONS
# ============================================================================

def calculate_force_distances(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pairwise Euclidean distances between all force patterns for each subject.
    
    This function computes all-pairs distances between force vectors within each subject's 
    dataset. This reveals how force patterns change across trials and blocks, providing 
    insight into motor learning and pattern stabilization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Force data with required columns:
        - 'SubNum': subject identifier
        - 'force_vector': np.array of force measurements for each trial
        - 'BN': block number
        - 'ET': execution time in milliseconds
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        - 'SubNum': subject identifier
        - 'Force_Distance': Euclidean distance between force patterns
        - 'Block_Distance': absolute difference in block numbers
        - 'ET_Distance': difference in execution times (ms)
    
    Examples
    --------
    >>> forces_correct = remove_error_trials(forces)
    >>> distances_df = calculate_force_distances(forces_correct)
    """
    distances = []

    for subind, subdata in tqdm(data.groupby('SubNum')):
        # Build 2D array of force_vectors (trials x force_dimensions)
        X = np.vstack(subdata['force_vector'].to_numpy())
        blocks = subdata['BN'].to_numpy()
        n = len(subdata)
        ets = subdata['ET'].to_numpy()
        
        # Upper triangular indices for distance calculation (avoid redundant comparisons)
        i, j = np.triu_indices(n, k=1)

        # Calculate pairwise distances
        force_dist = np.linalg.norm(X[i] - X[j], axis=1)
        block_dist = np.abs(blocks[i] - blocks[j])
        et_dist = ets[i] - ets[j]

        distances.append(pd.DataFrame({
            'SubNum': subind,
            'Force_Distance': force_dist,
            'Block_Distance': block_dist,
            'ET_Distance': et_dist
        }))

    # Concatenate all distances into a single DataFrame
    distances_df = pd.concat(distances, ignore_index=True)
    return distances_df




def kernel_smoother(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                   xi: np.ndarray, yi: np.ndarray, h: float = 0.1) -> np.ndarray:
    """
    Smooth scattered 2D data onto a regular grid using Gaussian kernel smoothing.
    
    This function performs kernel density estimation on scattered (x, y) points and 
    interpolates their associated z values onto a regular grid. Useful for visualizing 
    continuous surfaces from point clouds.
    
    Parameters
    ----------
    x : np.ndarray
        X-coordinates of data points (1D array).
    y : np.ndarray
        Y-coordinates of data points (1D array).
    z : np.ndarray
        Values at each (x, y) point to be smoothed (1D array).
    xi : np.ndarray
        X-coordinates of output grid points (2D meshgrid).
    yi : np.ndarray
        Y-coordinates of output grid points (2D meshgrid).
    h : float, optional
        Bandwidth (smoothing parameter). Larger values = more smoothing. Default is 0.1.
    
    Returns
    -------
    np.ndarray
        Smoothed z values interpolated onto the grid, same shape as xi and yi.
    
    Notes
    -----
    - Uses Gaussian kernel: w = exp(-(d/h)^2 / 2)
    - Points outside the smoothing bandwidth receive NaN values
    - Computational complexity: O(n_grid * n_points)
    
    Examples
    --------
    >>> xi = np.linspace(0, 10, 50)
    >>> yi = np.linspace(0, 10, 50)
    >>> XI, YI = np.meshgrid(xi, yi)
    >>> ZI = kernel_smoother(x, y, z, XI, YI, h=0.5)
    """
    # query points flattened
    xq = np.column_stack((xi.flatten(), yi.flatten()))
    x = np.column_stack((x, y))

    zout = np.zeros(len(xq))

    for j, q in enumerate(xq):
        # distances from query point to all data points
        d = np.linalg.norm(x - q, axis=1)
        # Gaussian kernel weights
        w = np.exp(-(d/h)**2 / 2)
        # normalized weighted average
        if np.sum(w) > 0:
            zout[j] = np.sum(w * z) / np.sum(w)
        else:
            zout[j] = np.nan
    return zout.reshape(xi.shape)


def kernel_smoother_1d(x: np.ndarray, y: np.ndarray, xi: np.ndarray, 
                       h: float = 0.1) -> np.ndarray:
    """
    Smooth 1D scattered data onto a 1D regular grid using Gaussian kernel smoothing.
    
    1D version of kernel_smoother for univariate smoothing. Performs kernel density 
    estimation on scattered (x) points and interpolates their y values onto a regular grid.
    
    Parameters
    ----------
    x : np.ndarray
        X-coordinates of data points (1D array).
    y : np.ndarray
        Values at each x point to be smoothed (1D array).
    xi : np.ndarray
        X-coordinates of output grid points (1D array).
    h : float, optional
        Bandwidth (smoothing parameter). Larger values = more smoothing. Default is 0.1.
    
    Returns
    -------
    np.ndarray
        Smoothed y values interpolated onto the grid, same shape as xi.
    
    Notes
    -----
    - Uses 1D Gaussian kernel: w = exp(-(d/h)^2 / 2)
    - Simpler and faster than 2D version for univariate data
    - Points outside the smoothing bandwidth receive NaN values
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2.1, 2.9, 4.2, 5.1])
    >>> xi = np.linspace(1, 5, 50)
    >>> yi = kernel_smoother_1d(x, y, xi, h=0.5)
    """
    xq = xi.flatten()
    zout = np.zeros(len(xq))

    for j, q in enumerate(xq):
        # distances from query point to all data points
        d = np.abs(x - q)
        # Gaussian kernel weights
        w = np.exp(-(d/h)**2 / 2)
        # normalized weighted average
        if np.sum(w) > 0:
            zout[j] = np.sum(w * y) / np.sum(w)
        else:
            zout[j] = np.nan
    return zout.reshape(xi.shape)

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

def set_figure_style(scale="1col"):
    """
    Set figure styling based on publication constraints.
    
    Parameters:
        scale (str): Scale of the figure, choose from "1col", "1.5col", "2col".
                     - "1col" for 8.5cm
                     - "1.5col" for 11.6cm
                     - "2col" for 17.6cm
    """
    # Define width options in cm
    widths = {"1col": 7.62, "1.5col": 11.6, "2col": 16.5}
    
    if scale not in widths:
        raise ValueError("Invalid scale. Choose from '1col', '1.5col', or '2col'.")
    
    # Convert width from cm to inches (1 cm = 0.393701 inches)
    width_in = widths[scale] * 0.393701
    
    # Set figure size (width, height)
    # Assuming height proportional to width (Golden Ratio)
    golden_ratio = (5**0.5 - 1) / 2
    rcParams["figure.figsize"] = (width_in, width_in * golden_ratio)
    
    # Set font sizes
    rcParams["font.size"] = 10  # General font size
    # rcParams["font.size"] = 20  # General font size
    rcParams["axes.titlesize"] = 12  # Figure title
    # rcParams["axes.titlesize"] = 26  # Figure title
    rcParams["axes.labelsize"] = 9  # Axis main label
    # rcParams["axes.labelsize"] = 22  # Axis main label
    rcParams["xtick.labelsize"] = 7  # Tick labels
    # rcParams["xtick.labelsize"] = 16  # Tick labels
    rcParams["ytick.labelsize"] = 7
    # rcParams["ytick.labelsize"] = 16
    rcParams["legend.fontsize"] = 8  # Legend entries
    # rcParams["legend.fontsize"] = 20  # Legend entries
    rcParams["figure.titleweight"] = "bold"
    
    # Set stroke width
    rcParams["axes.linewidth"] = 0.75
    # rcParams["axes.linewidth"] = 1.5

    # rcParams["lines.linewidth"] = 3
    
    rcParams["xtick.major.width"] = 0.75
    # rcParams["xtick.major.width"] = 1.5
    rcParams["ytick.major.width"] = 0.75
    # rcParams["ytick.major.width"] = 1.5

    
    # Subpanel lettering size
    rcParams["text.usetex"] = False  # Set to True if using LaTeX
    rcParams["axes.formatter.use_mathtext"] = True  # Math text for scientific notation

def add_subpanel_label(ax, label, fontsize=20, position=(-0.1, 1.05)):
    """
    Add a subpanel label (e.g., 'a', 'b') to a subplot.
    
    Parameters:
        ax (Axes): Matplotlib Axes object.
        label (str): The label text.
        fontsize (int): Font size for the label.
        position (tuple): Position of the label in axes coordinates.
    """
    ax.text(position[0], position[1], label, transform=ax.transAxes, 
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


### FORCE DISTANCE MATRIX PLOTTING
### for a given subject, plot the force distance matrix across all trials
### error trials are marked on x and y ticks
### red dashed lines separate days
### Execution times aligned to the bottom x-axis

def plot_force_movement_dynamics(data, subj, n_trials_per_day, n_days):

    subdata = data[data['SubNum'] == subj]
    force_vectors = np.stack(subdata['force_vector'].values)
    diff = force_vectors[:, np.newaxis, :] - force_vectors[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    vmin, vmax = np.percentile(distances, [5, 95])  # better visualization

    # Create subplots: top for heatmap, bottom for execution times
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [6, 1]})

    # Heatmap on top
    sns.heatmap(distances, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax1, cbar = False)
    for day in range(1, n_days):
        ax1.axvline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)
        ax1.axhline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)

    # Add colorbar to the right of the heatmap
    cbar_ax = fig.add_axes([1, 0.2, 0.01, 0.7])
    fig.colorbar(ax1.collections[0], cax=cbar_ax)

    # Plotting error trials on x and y ticks for heatmap
    error_trials = subdata[subdata['isError'] == 1]['N'] - 1  # zero-indexed
    ax1.set_xticks(error_trials)
    ax1.set_yticks(error_trials)
    ax1.set_xticklabels([''] * len(error_trials), color='black')  # Hide labels but keep ticks
    ax1.set_yticklabels([''] * len(error_trials), color='black')
    ax1.set_title(f'Subject {subj} Force Distance Matrix')
    ax1.set_xlabel('')  # Remove x-label for now
    ax1.set_ylabel('Trial Number')

    # Bottom plot for execution times
    trial_numbers = subdata['N'] - 1  # zero-indexed
    speeds = subdata['speed']
    ax2.plot(trial_numbers, speeds, color='black', linewidth=1)
    ax2.set_xlim(ax1.get_xlim())  # Align x-axis with heatmap
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Speed')

    # Add vertical lines for days on bottom plot
    for day in range(1, n_days):
        ax2.axvline(n_trials_per_day * day, color='red', linestyle='--', linewidth=0.7)

    # Mark error trials on bottom plot
    ax2.scatter(error_trials, speeds.iloc[error_trials], color='red', s=2)

    sns.despine()

    plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots for alignment
    plt.show()


def calc_neighbour_distances(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate force distances between consecutive (neighbouring) trials.
    
    Analyzes trial-by-trial learning by computing the Euclidean distance between 
    force patterns of consecutive trials. This reveals how force patterns evolve 
    within a block and whether execution time is correlated with force pattern changes.
    
    Parameters
    ----------
    data : pd.DataFrame
        Force data with required columns:
        - 'SubNum': subject identifier
        - 'force_vector': np.array of force measurements for each trial
        - 'trialPoints': -1 for error trials, ≥0 for correct trials
        - 'BN': block number
        - 'ET': execution time in milliseconds
        - 'N': trial number
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        - 'SubNum': subject identifier
        - 'Force_Distance': ||F(t+1) - F(t)||
        - 'first_ET': ET of trial t, centered to block mean
        - 'second_ET': ET of trial t+1, centered to block mean
        - 'count': count of comparisons
    
    Notes
    -----
    - Only considers consecutive trials within the same block
    - Excludes error trials from analysis
    - Execution times are centered to block mean for better interpretation
    
    Examples
    --------
    >>> distances_df = calc_neighbour_distances(forces)
    >>> plot_neighbour_distances(distances_df)
    """
    distances = []
    data_correct = data[data['trialPoints'] != -1]

    for subind, subdata in tqdm(data_correct.groupby('SubNum')):
        for block, block_data in subdata.groupby('BN'):
            mean_block_et = block_data['ET'].mean()
            for i in range(len(block_data) - 1):
                trial1 = block_data.iloc[i]
                trial2 = block_data.iloc[i + 1]
                if trial2['N'] - trial1['N'] == 1:  # only consider consecutive trials
                    # Calculate force distance
                    force_dist = np.linalg.norm(trial2['force_vector'] - trial1['force_vector'])
                    # Calculate ETs (centered to block mean)
                    first_ET = trial1['ET'] - mean_block_et
                    second_ET = trial2['ET'] - mean_block_et

                    distances.append({
                        'SubNum': subind,
                        'Force_Distance': force_dist,
                        'first_ET': first_ET,
                        'second_ET': second_ET
                    })

    # Concatenate all distances into a single DataFrame
    distances = pd.DataFrame(distances)
    distances['count'] = 1
    return distances





def plot_neighbour_distances(distances):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[6, 1, 5])
    
    ax_dens = fig.add_subplot(gs[0, 0])
    ax_cont_top = fig.add_subplot(gs[0, 1])
    ax_cont_bot = fig.add_subplot(gs[1, 1])
    ax_point = fig.add_subplot(gs[2, 0])

    # 1. Density Plot
    cmap = sns.color_palette('Blues', as_cmap=True)
    df_f = distances[(distances['first_ET'].between(-500, 500)) & (distances['second_ET'].between(-500, 500))]
    sns.kdeplot(data=df_f, x='first_ET', y='second_ET', fill=True, cmap=cmap, cbar=True, ax=ax_dens)
    ax_dens.plot([-500, 500], [-500, 500], color='black', linestyle='--', alpha=0.5)
    ax_dens.set_xlabel(r'$E_T -\overline{E}$ (ms)')
    ax_dens.set_ylabel(r'$E_{T+1} -\overline{E}$ (ms)')
    ax_dens.set_title('Density Plot')

    # 2. Contour Plot
    force_cutoff = df_f['Force_Distance'].quantile(0.95)
    df_cont = df_f[df_f['Force_Distance'] <= force_cutoff]
    
    xi = np.linspace(df_cont['first_ET'].min(), df_cont['first_ET'].max(), 100)
    yi = np.linspace(df_cont['second_ET'].min(), df_cont['second_ET'].max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = kernel_smoother(df_cont['first_ET'], df_cont['second_ET'], df_cont['Force_Distance'], XI, YI, h=50)
    
    m1 = ax_cont_top.contourf(XI, YI, ZI, cmap=cmap, levels=20)
    fig.colorbar(m1, ax=ax_cont_top)
    ax_cont_top.plot([-500, 500], [-500, 500], 'k--', alpha=0.5)
    ax_cont_top.set_title(r'|| $\vec{F}_{t+1} - \vec{F}_{t}$ ||')
    ax_cont_top.set_ylabel(r'$E_{T+1} -\overline{E}$ (ms)')

    # Marginal Contour
    xi_m, yi_m = np.meshgrid(xi, [0, 2])
    zi_m = kernel_smoother_1d(df_cont['first_ET'], df_cont['Force_Distance'], xi_m, h=50)
    m2 = ax_cont_bot.contourf(xi_m, yi_m, zi_m, cmap=cmap, levels=20)
    cbar2 = fig.colorbar(m2, ax=ax_cont_bot, aspect=3)
    cbar2.locator = ticker.MaxNLocator(nbins=3)
    ax_cont_bot.set_yticks([])
    ax_cont_bot.set_xlabel(r'$E_T -\overline{E}$ (ms)')
    ax_cont_bot.set_ylabel('marginal')

    # 3. Summary Pointplot
    ET_diff_bin_size = 200
    distances['abs_ET_diff'] = np.abs(distances['second_ET'] - distances['first_ET'])
    df_p = distances[distances['abs_ET_diff'] < 400].copy()
    df_p['is_pos_diff'] = (df_p['second_ET'] - df_p['first_ET']) > 0
    df_p['ET_diff_bin'] = df_p['abs_ET_diff'] // ET_diff_bin_size + 1
    
    grouped = df_p.groupby(['SubNum', 'ET_diff_bin', 'is_pos_diff']).agg({'Force_Distance': 'median'}).reset_index()
    sns.pointplot(data=grouped, x='ET_diff_bin', y='Force_Distance', hue='is_pos_diff', 
                  hue_order=[True, False], palette='colorblind', errorbar='se', ax=ax_point)
    
    ax_point.set_xlabel(r'|| $E_{t+1} - E_{t}$ ||')
    ax_point.set_ylabel(r'|| $\vec{F}_{t+1} - \vec{F}_{t}$ ||')
    handles, labels = ax_point.get_legend_handles_labels()
    ax_point.legend(handles, [r'$E_{t+1} > E_{t}$', r'$E_{t+1} < E_{t}$'], title='')
    
    bin_labels = [f'{(b-1)*ET_diff_bin_size}-{b*ET_diff_bin_size}' for b in sorted(grouped['ET_diff_bin'].unique())]
    ax_point.set_xticks(range(len(bin_labels)))
    ax_point.set_xticklabels(bin_labels)

    sns.despine(trim=True)
    plt.show()
    print("ANOVA results for neighbour distances:")
    print(AnovaRM(grouped, 'Force_Distance', 'SubNum', within=['ET_diff_bin', 'is_pos_diff']).fit())


def calc_triplet_distances(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate force distances between first and third trials in a triplet.
    
    Analyzes the effect of intervening trial. Compares force patterns 
    of trial t and trial t+2, while tracking whether trial t+1 was 
    an error or correct. This helps determine if errors effect learning dynamics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Force data with required columns:
        - 'SubNum': subject identifier
        - 'force_vector': np.array of force measurements for each trial
        - 'trialPoints': -1 for error trials, ≥0 for correct trials
        - 'BN': block number
        - 'ET': execution time in milliseconds
        - 'N': trial number
    
    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        - 'SubNum': subject identifier
        - 'Force_Distance': ||F(t+2) - F(t)||
        - 'ET_Distance': |ET(t+2) - ET(t)|
        - 'is_middle_error': True if trial t+1 was an error, False otherwise
        - 'count': count of comparisons
    
    Notes
    -----
    - Only considers triplets of consecutive trials
    - Both first and last trials must be correct for inclusion
    
    Examples
    --------
    >>> distances_df = calc_triplet_distances(forces)
    >>> plot_triplet_distances(distances_df)
    """
    distances = []

    for subind, subdata in tqdm(data.groupby('SubNum')):
        for block, block_data in subdata.groupby('BN'):
            for i in range(len(block_data) - 2):
                trial1 = block_data.iloc[i]
                trial2 = block_data.iloc[i + 1]
                trial3 = block_data.iloc[i + 2]
                if (trial2['N'] - trial1['N'] == 1) and (trial3['N'] - trial2['N'] == 1):  # only consider triplets of consecutive trials
                    if (trial1['trialPoints'] != -1) and (trial3['trialPoints'] != -1): # if the first and last are correct
                        # Calculate force distance
                        force_dist = np.linalg.norm(trial3['force_vector'] - trial1['force_vector'])
                        # Calculate ET distance
                        et_dist = trial3['ET'] - trial1['ET']
                        error_in_the_middle = (trial2['trialPoints'] == -1)

                        distances.append({
                            'SubNum': subind,
                            'Force_Distance': force_dist,
                            'ET_Distance': np.abs(et_dist),
                            'is_middle_error': error_in_the_middle 
                        })

    # Concatenate all distances into a single DataFrame
    distances = pd.DataFrame(distances)
    distances['count'] = 1
    return distances


def plot_triplet_distances(distances):
    ET_diff_bin_size = 100
    filtered_distances = distances[distances['ET_Distance'] < 200].copy()
    filtered_distances['ET_diff_bin'] = filtered_distances['ET_Distance'] // ET_diff_bin_size + 1
    grouped_distances = filtered_distances.groupby(['SubNum', 'ET_diff_bin', 'is_middle_error']).agg({
        'Force_Distance': 'median',
        'count': 'sum'
    }).reset_index()

    # Fix hue order so legend ordering is stable
    hue_order = [False, True]
    ax = sns.pointplot(
        data=grouped_distances,
        x='ET_diff_bin',
        y='Force_Distance',
        errorbar='se',
        hue = 'is_middle_error',
        hue_order=hue_order,
        palette='colorblind'
    )
    plt.xlabel(r' $|E_{t+1} - E_{t-1}|$')
    plt.ylabel(r'|| $\vec{F}_{t+1} - \vec{F}_{t-1}$ ||')

    handles, labels = ax.get_legend_handles_labels()
    label_map = {'True': 't is error', 'False': 't is correct'}
    ax.legend(handles, [label_map[l] for l in labels], title='', loc='upper right', bbox_to_anchor=(1.3, 1))


    # Create bin labels showing the actual intervals
    bin_labels = []
    for bin_num in sorted(grouped_distances['ET_diff_bin'].unique()):
        lower = (bin_num - 1) * ET_diff_bin_size
        upper = bin_num * ET_diff_bin_size
        bin_labels.append(f'{lower}-{upper}')
    plt.xticks(range(len(bin_labels)), bin_labels)

    sns.despine(trim=True)

    print("ANOVA results for for triplet distances::")
    print(AnovaRM(grouped_distances, 'Force_Distance', 'SubNum', within=['ET_diff_bin', 'is_middle_error']).fit())



def plot_speed_clamped_ETs(subjs):
    set_figure_style(scale='2col')

    subjs_correct = remove_error_trials(subjs).copy()
    condition_order = ['Unclamped', 'Clamped', 'Verbal']
    group_palette = {
        'Unclamped': sns.color_palette('colorblind')[0],
        'Clamped': sns.color_palette('colorblind')[1],
        'Verbal': 'gray'
    }

    # discard posttest_clamped_speed
    subjs_correct = subjs_correct[subjs_correct['phase'] != 'posttest_clamped_speed']

    # --- ADJUSTABLE PARAMETERS ---
    phase_width = {
        'pretest': 1.5,               # stretch/shrink inner-block spacing
        'training': 1.0,             
        'posttest_full_speed': 1.5,   
        'posttest_clamped_speed': 1.0 
    }

    phase_gap = 0  # Space between pretest, training, and posttest
    day_gap = 3.0    # Space between Day 1 and Day 2
    # -----------------------------

    # 1) Build a unified map for every (day, BN) to a warped x-coordinate
    bn_ordered = (
        subjs_correct[['day', 'phase', 'BN']]
        .drop_duplicates()
        .sort_values(['day', 'BN'])
        .reset_index(drop=True)
    )

    x_coords = []
    current_x = 0.0

    for i, row in bn_ordered.iterrows():
        if i == 0:
            current_x = 0.0
        else:
            prev_row = bn_ordered.iloc[i-1]
            w = phase_width.get(row['phase'], 1.0)
            
            # Determine the spacing to add based on transitions
            if row['day'] != prev_row['day']:
                current_x += w + day_gap        # Transition to new day
            elif row['phase'] != prev_row['phase']:
                current_x += w + phase_gap      # Transition to new phase
            else:
                current_x += w                  # Continuing same phase
                
        x_coords.append(current_x)

    # Map the calculated coordinates back to the main dataframe
    bn_ordered['x_plot'] = x_coords
    # Using (day, BN) as a tuple key in case BN resets to 0 on day 2
    bn_to_x = dict(zip(zip(bn_ordered['day'], bn_ordered['BN']), bn_ordered['x_plot']))

    # Apply the x_plot mapping to the full dataset
    subjs_correct['x_plot'] = subjs_correct.set_index(['day', 'BN']).index.map(bn_to_x)


    # 2) Plotting Loop
    ax = plt.gca()
    legend_drawn = False

    for day, day_data in subjs_correct.groupby('day'):
        phases = ['pretest', 'training', 'posttest_full_speed']
        
        for phase_name in phases:
            data = day_data[day_data['phase'] == phase_name]
            if data.empty: continue
            
            # Group by including the newly mapped x_plot
            g = data.groupby(['SubNum', 'group', 'BN', 'x_plot'], as_index=False)['ET'].median()
            
            # Plot segment
            sns.lineplot(
                data=g, x='x_plot', y='ET', hue='group',
                palette=group_palette, hue_order=condition_order,
                errorbar='se', 
                legend=not legend_drawn, # Only draw the legend once
                ax=ax
            )
            legend_drawn = True

    # --- CLEAN UP AXES ---
    ax.set_xlabel('Blocks (Adjusted Spacing)')
    ax.set_ylabel('Execution Time (ET)')

    # Hide standard warped ticks since the spacing is custom
    ax.set_xticks([]) 

    sns.despine(trim=True)
    plt.tight_layout()
