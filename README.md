# EMG_RMVAE
Code associated with the project "An Analysis Of The Dimensionality Of Motor Unit Control In The Vastus Lateralis And Gastrocnemius Medialis Using Latent Factor Models".

The models in this repository are models of rigid recruitment of motor units.

As mentioned in the associated report, the models in this repository have no predictive capabilities.
They function to overfit on a set of motor unit firing rates and get every opportunity to reconstruct their input data as accurately as possible.
In the scenarios where they can reconstruct the motor unit firing rates, the state space motor unit firing rates follows either a linear on non-linear monotonic manifold.
- If they do, then rigid recruitment is observed and we find that the population of motor units follows rigid control.
- If not, then something else is at play. The dimensionality of the latent space and (non-)linearity of the model used may shed some light on the flavour of the else.
