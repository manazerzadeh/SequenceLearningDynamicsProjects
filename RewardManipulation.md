# Reward Manipulation Effect on Trial-by-Trial Force Pattern Dynamics

## Overview
Investigation of how external reward (points feedback) based on execution time performance affects motor learning at trial-by-trial resolution. The critical manipulation is a **randomized point system** that decorrelates speed performance (fast vs slow) from reward (points earned).

## Experimental Design

**Participants**: N = 11 subjects  
**Duration**: 4 days of training  
**Task**: Single 7-keystroke sequence
**Accuracy requirement**: Minimum 80% accuracy maintained throughout  
**Session structure**: 4 blocks per day × 40 trials per block = 160 trials/day

## Reward Zone Manipulation

Each trial's execution time (ET) is classified into zones based on the *previous block's* ET distribution:

### Zone Classification (Based on Percentiles)
- **Zone 4** (0-10th percentile, fastest): 3 points (deterministic)
- **Zone 3** (10-45th percentile): 1 or 3 points (50% random)
- **Zone 2** (45-80th percentile): 0 or 1 point (50% random)
- **Zone 1** (80-100th percentile, slowest): 0 points (deterministic)

## Data Structure

### Main Data Files
- `Reward_Manipulation_Forces.csv`: Force Patterns
- `Reward_Manipulation_Subjs.csv`: Subject-level behavioral data

### Key Columns
- `SubNum`: Subject identifier (1-11)
- `zone`: ET percentile zone (1-4, based on previous block distribution)
- `points`: Points earned (0-3)
- `force_vector`: Concatenated force measurements
- `ET`: Execution time (milliseconds)
- `BN`: Block number
- `day`: Day of practice (1-4)

