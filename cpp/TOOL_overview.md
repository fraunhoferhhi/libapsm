# NOMA C++ Overview

NOMA on GPUs library toolset overview is given here.

## NOMA offline data format

We have defined the following offline data content.
This is used for scripting as well as for unit testing and debugging.

### reference data or random seed value

This data is used in any tool you find the -r or --reference-data option e.g. in NOMA\_demo and NOMA\_cli tool as input or in NOMA\_dataset as output.

| Value | Format   | Description     | Comment                |
|-------|----------|-----------------|------------------------|
| 2     | int32    | Num keys        | total keys             |
| 8     | int32    | Len name 1      | key 1 name length      |
| [...] | char[]   | [mod_data]      | key 1 name             |
| 12    | int32    | Len name 2      | key 2 name length      |
| [...] | char[]   | [mod_training]  | key 2 name             |
| 3     | int32    | Dimensions      | num dims for key 1     |
| 3840  | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 122880    | 3840 x 16 x 2          |
| 3     | int32    | Dimensions      | num dims for key 2     |
| 685   | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 21920     | 685 x 16 x 2           |

Example:

- cpp/data/offline/tx/NOMA_signals_qam16_complex.bin

```console
mod_data: [3840, 8, 2]
mod_training: [685, 8, 2]
```

### synchronized data

This data is used in any tool you find the -s or --synchronized-data option e.g. in NOMA\_tool as input or in NOMA\_convert as output.

| Value | Format   | Description     | Comment                |
|-------|----------|-----------------|------------------------|
| 2     | int32    | Num keys        | total keys             |
| 13    | int32    | Len name 1      | key 1 name length      |
| [...] | char[]   | [rxSigTraining] | key 1 name             |
| 9     | int32    | Len name 2      | key 2 name length      |
| [...] | char[]   | [rxSigData]     | key 2 name             |
| 3     | int32    | Dimensions      | num dims for key 1     |
| 685   | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 21920     | 685 x 16 x 2           |
| 3     | int32    | Dimensions      | num dims for key 2     |
| 3840  | int32    | Dim 1           | dim 1 length           |
| 16    | int32    | Dim 2           | dim 2 length           |
| 2     | int32    | Dim 3           | dim 3 length           |
| [...] | float[]  | Data: 122880    | 3840 x 16 x 2          |

- cpp/data/offline/rx/time/rxData_QAM16_alltx_converted.bin
- cpp/data/offline/rx/ofdm/rxData_QAM16_alltx_converted.bin

```console
rxSigTraining: [685, 16, 2]
rxSigData: [3840, 16, 2]
```

## NOMA\_detect

This tool runs the training and detection algorithms. The algorithm configuration is passed in JSON format either as string or in a file.

```
Optional arguments:
-h  --help                       shows help message and exits [default: false]
-v  --version                    prints version information and exits [default: false]
-s  --synchronized-data          synchronized data file containing received training and data signals [required]
-r  --reference-data             reference data file
-rs --random-seed                random seed for generation of data (this overrides the --reference-data argument)
-d  --detected-data              detected (estimated) data file
-cf --config-file                filename for JSON file containing the algorithm configuration
-cs --config-string              JSON string containing the algorithm configuration [default: "{ "algorithm": { "otype": "APSM" } }"]
-o  --output-mode                print mode for error metric (BER = bit errors, SER = symbol errors, EVM = Error Vector Magnitude, scripting = disable all output except "ber,ser,evm") [default: "SER"]
-th --train-history-file         filename for training history CSV-file (test set will be used as validation set) [default: ""]
-p  --parallel-processing        process all users in parallel [default: false]
```

## JSON config format

```json
{
    "algorithm": { ... },  // configuration for detection algorithm (see below)
    "modulation": "QPSK",  // modulation used for data generation and error computation (valid values: off, bpsk, qpsk, qam16, qam64, qam256, qam1024, qam4096, qam16384)
    "user": 0,             // user, which should be decoded
    "antennas": { ... }    // selection of antennas (see below)
}
```

### Algorithms

#### LLS

Linear least squares algorithm

```json
{
    "otype": "LLS"
}
```

#### APSM

Adaptive projected subgradient method

```json
{
    "otype": "APSM",
    "llsInitialization": false,      // should the linear weights be initialized with the LLS solution?
    "trainingShuffle": false,        // should the training data be shuffled?
    "gaussianKernelWeight": 0.5,     // Gaussian Kernel Weight
    "gaussianKernelVariance": 0.05,  // Gaussian Kernel Variance
    "windowSize": 20,                // number of past samples to reuse for each training iteration
    "sampleStep": 2,                 // number of samples the training window will advance in each iteration
    "trainPasses": 1,                // number of training passes over the training data
    "startWithFullWindow": false,    // if true skip the smaller windows at the beginning of training
    "dictionarySize": 685,           // maximum number of (complex) basis vectors to be saved in the dictionary
    "normConstraint": 0.,            // norm constraint to use for dictionary sparsification (set to 0 to disable)
    "eB": 0.001,                     // hyperslab width
    "detectVersion": "balanced"      // which (optimized) implementation to use (valid values: original, oldfast, shmem, balanced)

}
```

### Antennas

#### All Antennas

```json
{
    "otype": "all"
}
```

#### Direct Pattern

```json
{
    "otype": "pattern",
    "pattern": "0011100110111010"  // directly specify which antennas should be used (should contain 16 chars which can be 1 (on) or 0(off))
}
```

#### Equidistant

```json
{
    "otype": "equidistant",
    "number": 16 // number of antennas which will be chosen equidistantly
}
```

#### Random

```json
{
    "otype": "random",
    "number": 16 // number of antennas which will be chosen randomly
}
```

#### First

```json
{
    "otype": "first",
    "number": 16 // number of antennas which will be chosen from the beginning
}
```

## Examples

Run the NOMA algorithm:

```bash
cd build

bin/NOMA_detect -r dataset/offline/tx/NOMA_signals_qpsk_complex.bin -s dataset/offline/rx/time/rxData_QPSK_alltx_converted.bin -cs '{"algorithm": {"otype":"APSM", "windowSize": 80}, "modulation":"QPSK", "antennas": {"otype": "equidistant", "number": 8}, "user": 1}'
bin/NOMA_detect -r dataset/offline/tx/NOMA_signals_qpsk_complex.bin -s dataset/offline/rx/ofdm/rxData_QPSK_alltx_converted.bin -cs '{"algorithm": {"otype":"APSM", "windowSize": 80}, "modulation":"QPSK", "antennas": {"otype": "equidistant", "number": 8}, "user": 1}'
```
