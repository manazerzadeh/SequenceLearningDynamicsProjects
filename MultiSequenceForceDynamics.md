# Ariani's Repetition Experiment (Ariani et al. 2020)

## Overview
Investigation of how force patterns change during repeated sequence execution.

## Experimental Design

**Participants**: N = 38 subjects  
**Duration**: 2 days of training  
**Sequences**: 8 different sequences of 4 key presses each  
**Session structure**: 12 blocks per day × 50 trials per block = 600 trials/day

**Repetition Design**: Within each block, subjects practice each sequence in *bouts* of 1, 2, 3, 4, or 5 repetitions. This allows analysis of:
- **Within-bout repetition learning**: How force patterns change across consecutive executions
- **Between-bout transitions**: How bout boundaries change learned patterns

## Data Structure

### Main Data Files
- `Ariani_Multi_Seq_Forces.csv`: Force Patterns
- `Ariani_Multi_Seq_Subjs.csv`: Subject-level behavioral data (execution times, errors)

### Key Columns
- `SubNum`: Subject identifier (1-38)
- `seqNum`: Sequence number (1-8)
- `repNum`: Repetition number within bout (1-5)
- `repType`: Bout length
- `force_vector`: Concatenated force array for the trial
- `ET`: Execution time (milliseconds)
- `BN`: Block number
- `isError`: 0=correct, 1=error

