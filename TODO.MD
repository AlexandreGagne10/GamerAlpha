# TODO

## Immediate Fixes
- Replace the `policy_loss` calculation in `train.py` to use the MCTS policy
  distribution (`targets_policy`) rather than just the chosen action.
- Add docstrings and type hints to all public functions and classes
  (`train.py` and files under `muzero/`).
- Update tests in `tests/` so that they use the new `select_action` implementation
  and work on both CPU and CUDA if available.
- Provide a setup script or instructions to install dependencies like `torch`
  and `gym` to make running tests smoother.

## Future Improvements
- Implement a more sophisticated MuZero replay buffer with priority sampling and
  support for larger datasets.
- ~~Add support for saving/loading model checkpoints during training.~~ Implemented.
- Expand the test suite to cover the full training loop and edge cases.
- Improve command line interface with more configuration options (learning rate,
  discount factor, network size, etc.).
- Add continuous integration configuration to automatically run tests.
- Include examples or notebooks demonstrating usage and training results.
