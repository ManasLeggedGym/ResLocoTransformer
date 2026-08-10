# Project TODO
## Ablations
Ablations are meant to be experiments that justify each and every architectural choice so that one can conretely say why a certain choice was made. The current approach is a two stage training where we use 

### Architecture Ablations

**Goal:** decide whether visual tokens and attention residuals (AttnRes) are worth their extra compute on rough terrain.

- **MLP:** state-only baseline; no visual terrain features.
- **Visual pool:** uses the visual encoder, then averages its tokens; no attention layers.
- **AttnRes:** uses the same visual encoder plus attention-residual layers. This tests whether attention itself adds value beyond visual features.


For the above you will need to modify the config json files in `configs/` folder.

Things to keep in mind:
- [ ] Logging should be nicely written and done to prevent loss of progress.
- [ ] Checkpoints need to be saved at regular intervals.
- [ ] Do not be fooled by just "high rewards" or low loss values - the final gait must be practical without the Quadruped dragging across the floor or something.
