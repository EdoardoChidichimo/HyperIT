Usage
============

HyperIT uses a Class/OOP framework, allowing multiple instances of HyperIT objects (instantiated with different data). MI, TE, and $\Phi\text{ID}$ atoms can be computed by calling the following functions:

.. code-block:: python
    
    from hyperit import HyperIT

    # Only needs to be called once, pass file location of local infodynamics.jar
    HyperIT.setup_JVM(jarLocation)

    # Gather your data here ...

    # Create instance
    it = HyperIT(data1, data2, channel_names, sfreq, freq_bands, verbose)

    # ROIs can be specified and then reset back to default
    it.roi(roi_list)
    it.reset_roi()

    # Calculate Mutual Information and Transfer Entropy
    mi = it.compute_mi(estimator_type='kernel', calc_sigstats=True, vis=True)
    te_xy, te_yx = it.compute_te(estimator_type='gaussian', calc_sigstats=True, vis=True)

    # Calculate Integrated Information Decomposition
    atoms = it.compute_atoms(tau=5, redundancy='mmi', vis=True)