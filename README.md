# NOMA APSM CUDA library

## Compile library

To download submodules and compile please run the init.sh script.

```bash
./init.sh
```

## Run example command line tool

```bash
cd build

bin/NOMA_detect -r dataset/offline/tx/NOMA_signals_qpsk_complex.bin -s dataset/offline/rx/time/rxData_QPSK_alltx_converted.bin -cs '{"algorithm": {"otype":"APSM", "windowSize": 80}, "modulation":"QPSK", "antennas": {"otype": "equidistant", "number": 8}, "user": 1}'
bin/NOMA_detect -r dataset/offline/tx/NOMA_signals_qpsk_complex.bin -s dataset/offline/rx/ofdm/rxData_QPSK_alltx_converted.bin -cs '{"algorithm": {"otype":"APSM", "windowSize": 80}, "modulation":"QPSK", "antennas": {"otype": "equidistant", "number": 8}, "user": 1}'
```

For more information about input file format and tool parameters please refer to [tool overview](cpp/TOOL_overview.md) file.

## License and Citation

APSM library license is specified, as found in the [LICENSE](LICENSE.md) file.

If you use this software, please cite it as:

```bibtex
@article{libapsm,
    title={GPU-Accelerated Partially Linear Multiuser Detection for 5G and Beyond URLLC Systems}, 
    author={Mehlhose, Matthias and Marcus, Guillermo and Schäufele, Daniel and Awan, Daniyal Amir and Binder, Nikolaus and Kasparick, Martin and Cavalcante, Renato L. G. and Stañczak, Sławomir and Keller, Alexander},
    journal={IEEE Access}, 
    year={2022},
    volume={10},
    number={},
    pages={70937-70946},
    doi={10.1109/ACCESS.2022.3187040}
}
```
