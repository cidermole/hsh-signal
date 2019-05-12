# HeartShield signal processing

This library was written during research on heart biosignals.

## FIR filters
Most useful is probably the part with FIR filters, based on gnuradio's implementation of the Window Method for FIR Filter Design.

See `hsh_signal/signal.py` for methods (low-pass, high-pass, etc.) directly applicable to an array of evenly-sampled signal values, or `hsh_signal/filter.py` for block-connectable filters intended for real-time applications.

## AliveCor decoder
The file `hsh_signal/alivecor.py` implements a decoder for AliveCor's Kardia EKG (electrocardiograph). This device captures the raw electrical signal, which is then frequency-modulated onto an 18.8 kHz audio carrier that is transmitted to the microphone in a phone/tablet.

Demodulation of the audio signal can be achieved via a phase-locked loop, with subsequent low-pass filter to remove high-frequency electrical noise, and a band-reject filter to remove 50 Hz mains noise. Note that for the US, you may need to change this filter to 60 Hz.

-- David <git@abanbytes.eu>

