To run the snapshots, copy the corresponding src and configs dir to the root of the repository.

- `muon_dropout` dropout in the WM
- `muon_encoder_decoder` muon only on the encoder + decoder, corresponds to "The One Muon Setup That Works"
- `reconstruction_free` various reconstruction-free configs
- `reconstruction_free_ablations` removing the from-h predictor and sigreg
- `varying_gamma` the experiment with changing gamma inspired by BBF 